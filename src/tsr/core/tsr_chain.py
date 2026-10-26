# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import logging
import math
from dataclasses import dataclass
from functools import reduce

import numpy

from . import _chain_solver
from .tsr import FRAME_ATOL, NANBW, TSR
from .utils import CHAIN_TOLERANCE, geodesic_distance, wrap_to_interval

logger = logging.getLogger(__name__)


def _require_int(value, name, *, minimum):
    """Reject non-integers, booleans, and values below ``minimum`` (#89)."""
    if isinstance(value, bool) or not isinstance(value, (int, numpy.integer)):
        raise ValueError(f"{name} must be a non-boolean integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class ChainSample:
    """A sampled chain pose together with the coordinates that constructed it.

    ``coordinates`` (shape ``(n, 6)``) is a *constructive witness*: composing it
    forward reproduces ``pose`` exactly, so membership needs no optimizer (see
    :meth:`TSRChain.validate_witness`).
    """

    pose: numpy.ndarray
    coordinates: numpy.ndarray


@dataclass(frozen=True)
class ChainSolveResult:
    """Result of an inverse chain solve (:meth:`TSRChain.solve`).

    ``status`` is ``"satisfied"`` when a witness within tolerance was found, or
    ``"not_found"`` when the bounded numerical search did not find one.
    ``"not_found"`` is **not** a certificate of non-membership — a finite set of
    local solves cannot prove global infeasibility for a rotation-rich chain.
    ``residual`` is the geodesic residual at ``coordinates`` (an upper bound on
    the true minimum). ``starts`` is the number of optimizer starts actually run
    (0 on an exact/fast path) and ``nfev`` the total smooth-residual objective
    evaluations across them — ``starts <= max_starts`` and ``nfev <= max_nfev``
    strictly (see :meth:`TSRChain.solve`).
    """

    status: str
    coordinates: numpy.ndarray
    residual: float
    nfev: int
    starts: int


class TSRChain:
    """
    A sequence of composed TSRs.

    **Composition rule** (Berenson et al. 2011 §5.1). The first link is placed by its
    own ``T0_w``. Every later link is placed *by the chain*: its frame **is** the
    previous link's end frame, ``C_{i-1}.T0_w · Tw(ξ_{i-1}) · C_{i-1}.Tw_e``. So a later
    link's own ``T0_w`` is never read, and :meth:`append` refuses a non-identity one
    rather than ignoring it (#166) — a dropped offset is a pose that is quietly wrong by
    exactly that offset, which is worse to debug than a rejected argument.

    **Where an offset belongs.** To sit a later link's region a fixed distance from the
    previous link's end frame, put that offset in the **previous** link's ``Tw_e``, which
    is the thing that moves the end frame the next link hangs off.

    Useful for articulated constraints: a door hinge with free yaw, then a handle a fixed
    distance along the door.
    """

    def __init__(self, TSR=None, TSRs=None, tsr=None):
        """
        Create a TSR chain from one or more TSRs.

        @param TSR a single TSR to use in this TSR chain
        @param TSRs a list of TSRs to use in this TSR chain
        @param tsr alias for TSR parameter
        """
        self.TSRs = []

        # Handle both TSR and tsr parameters
        single_tsr = TSR if TSR is not None else tsr
        if single_tsr is not None:
            self.append(single_tsr)
        if TSRs is not None:
            for tsr_item in TSRs:
                self.append(tsr_item)

    def append(self, tsr):
        """Add a link to the end of the chain.

        Raises ``ValueError`` if a link after the first carries a ``T0_w`` the chain will
        not use. See the composition rule in the class docstring.
        """
        if self.TSRs and not numpy.allclose(tsr.T0_w, numpy.eye(4), rtol=0.0, atol=FRAME_ATOL):
            raise ValueError(
                "a TSR after the first in a chain is positioned by the chain: its frame "
                "is the previous link's end frame, so its T0_w is never read and must be "
                f"the identity (got a translation of {numpy.round(tsr.T0_w[:3, 3], 6).tolist()} m "
                f"and rotation trace {numpy.trace(tsr.T0_w[:3, :3]):.6g}). Put the offset in the "
                "PREVIOUS link's Tw_e, which is what moves the end frame this link hangs off."
            )
        self.TSRs.append(tsr)

    def to_dict(self):
        """Convert TSR chain to a python dict."""
        return {
            "tsrs": [tsr.to_dict() for tsr in self.TSRs],
        }

    @staticmethod
    def from_dict(x):
        """Construct a TSR chain from a python dict."""
        return TSRChain(
            TSRs=[TSR.from_dict(tsr) for tsr in x["tsrs"]],
        )

    def to_json(self):
        """Convert this TSR chain to a JSON string."""
        import json

        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR chain from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json

        x_dict = json.loads(x, *args, **kw_args)
        return TSRChain.from_dict(x_dict)

    def to_yaml(self):
        """Convert this TSR chain to a YAML string."""
        import yaml

        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x):
        """Construct a TSR chain from a YAML string."""
        import yaml

        x_dict = yaml.safe_load(x)
        return TSRChain.from_dict(x_dict)

    def is_valid(self, xyzrpy_list, ignoreNAN=False):
        """
        Checks if a xyzrpy list is a valid sample from the TSR.
        @param xyzrpy_list a list of xyzrpy values
        @param ignoreNAN (optional, defaults to False) ignore NaN xyzrpy
        @return a list of 6x1 vector of True if bound is valid and False if not
        """

        if len(self.TSRs) == 0:
            raise ValueError("Cannot validate against empty TSR chain!")

        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError("Sample must be of equal length to TSR chain!")

        check = []
        for idx in range(len(self.TSRs)):
            check.append(self.TSRs[idx].is_valid(xyzrpy_list[idx], ignoreNAN))

        return check

    def to_transform(self, xyzrpy_list):
        """
        Converts a xyzrpy list into an end-effector transform.

        This implements TSR chain composition as described in Section 5.1 of
        Berenson et al. 2011:
            C_i.T0_w = (C_{i-1}.T0_w) * (C_{i-1}.Tw_sample) * (C_{i-1}.Tw_e)

        The final transform is: T0_sample = Cn.T0_w * Cn.Tw_sample * Cn.Tw_e

        @param xyzrpy_list  a list of xyzrpy values, one per TSR in the chain
        @return trans       4x4 transform
        """
        if len(self.TSRs) == 0:
            raise ValueError("Cannot compute transform for empty TSR chain")

        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError(f"xyzrpy_list length ({len(xyzrpy_list)}) must match number of TSRs ({len(self.TSRs)})")

        # Map coordinates into each component's continuous _Bw_cont chart (wrap
        # rotations, then clip) via the shared helper, so the forward path and the
        # inverse optimiser-start construction canonicalise identically (#87, #90).
        continuous = self._to_continuous([numpy.asarray(x, dtype=float) for x in xyzrpy_list])

        # Compute the chained transform WITHOUT modifying original TSR objects,
        # starting from the first TSR's T0_w.
        T0_w_current = self.TSRs[0].T0_w
        for idx, tsr in enumerate(self.TSRs):
            Tw_sample = TSR.xyzrpy_to_trans(continuous[idx])
            T0_w_current = reduce(numpy.dot, [T0_w_current, Tw_sample, tsr.Tw_e])

        return T0_w_current

    def _to_continuous(self, coordinates):
        """Map public ``(n, 6)`` coordinates into the components' continuous charts.

        Rotations are periodic: each RPY coordinate is wrapped into its component's
        ``_Bw_cont`` interval with :func:`wrap_to_interval` (so a valid coordinate
        expressed in ``[-pi, pi]``, e.g. ``-3.0`` for a wrapping interval
        ``[3pi/4, -3pi/4]``, becomes its in-chart equivalent rather than being
        clipped to an unrelated boundary). Translations are not periodic and are
        only clipped. All six are clipped to bounds *after* wrapping (values
        slightly outside bounds are normal during the optimiser's line search).
        Shared by :meth:`to_transform` and the inverse solver's start construction
        so the forward and inverse paths cannot drift apart (#87, #90).
        """
        coordinates = numpy.asarray(coordinates, dtype=float)
        out = numpy.empty((len(self.TSRs), 6))
        for idx, tsr in enumerate(self.TSRs):
            Bw = tsr._Bw_cont
            row = coordinates[idx].copy()
            row[3:6] = wrap_to_interval(row[3:6], lower=Bw[3:6, 0])
            out[idx] = numpy.clip(row, Bw[:, 0], Bw[:, 1])
        return out

    def sample_xyzrpy(self, xyzrpy_list=None, rng=None):
        """
        Samples from Bw to generate a list of xyzrpy samples
        Can specify some values optionally as NaN.

        @param xyzrpy_list   (optional) a list of Bw with float('nan') for
                        dimensions to sample uniformly.
        @param rng      (optional) a numpy.random.Generator for reproducible
                        sampling. Defaults to the global numpy RNG when None.
        @return sample  a list of sampled xyzrpy
        """

        if xyzrpy_list is None:
            xyzrpy_list = [NANBW] * len(self.TSRs)

        sample = []
        for idx in range(len(self.TSRs)):
            sample.append(self.TSRs[idx].sample_xyzrpy(xyzrpy_list[idx], rng=rng))

        return sample

    def sample(self, xyzrpy_list=None, rng=None):
        """
        Samples from the Bw chain to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param xyzrpy_list   (optional) a list of xyzrpy with float('nan') for
                             dimensions to sample uniformly.
        @param rng      (optional) a numpy.random.Generator for reproducible
                        sampling. Defaults to the global numpy RNG when None.
        @return T0_w         4x4 transform
        """
        return self.to_transform(self.sample_xyzrpy(xyzrpy_list, rng=rng))

    # Default TOTAL inverse-solve budget (#85, #89): at most _DEFAULT_MAX_STARTS
    # optimiser starts, sharing at most _DEFAULT_MAX_NFEV objective evaluations
    # across the whole solve. The default start count matches the pre-#89 schedule
    # (midpoint + two opposite corners + 8 seeded interior points), and the default
    # per-start slice (2200 / 11 = 200) matches the pre-#89 per-start cap, so
    # ordinary cold queries do no more work than before.
    _DEFAULT_MAX_STARTS = 11
    _DEFAULT_MAX_NFEV = 2200

    def sample_with_witness(self, xyzrpy_list=None, rng=None):
        """Sample a pose and return it with the coordinates that constructed it.

        Same sampling and one forward composition as :meth:`sample`; performs no
        optimisation. The returned coordinates are a constructive witness, so
        membership is checkable exactly via :meth:`validate_witness` without an
        inverse solve (#85).
        """
        coords = numpy.array(self.sample_xyzrpy(xyzrpy_list, rng=rng))
        return ChainSample(pose=self.to_transform(coords), coordinates=coords)

    def _witness_residual(self, trans, coordinates):
        """Geodesic residual for ``coordinates`` if they are a valid witness, else None.

        One bounds check and one forward composition; no optimiser, no SciPy. An
        empty chain (shape ``(0, 6)``) has no witness and returns None (#87, #88).
        """
        coordinates = numpy.asarray(coordinates, dtype=float)
        n = len(self.TSRs)
        if n == 0 or coordinates.shape != (n, 6):
            return None
        for i in range(n):
            if not all(self.TSRs[i].is_valid(coordinates[i])):
                return None
        return float(geodesic_distance(self.to_transform(coordinates), trans))

    def validate_witness(self, trans, coordinates, tolerance=CHAIN_TOLERANCE):
        """Exact positive certificate: are ``coordinates`` a valid witness for ``trans``?

        Validates the coordinates against each component's bounds and recomposes
        the pose once, returning True iff the recomposition matches ``trans``
        within ``tolerance``. Uses no optimiser and no SciPy (#85). An empty chain
        has no witness, so this returns False rather than raising (#87).
        """
        residual = self._witness_residual(trans, coordinates)
        return residual is not None and residual < tolerance

    def solve(self, trans, initial_guess=None, max_starts=None, max_nfev=None, tolerance=CHAIN_TOLERANCE):
        """Inverse chain solve: find coordinates whose composition matches ``trans``.

        Returns a :class:`ChainSolveResult`. A valid ``initial_guess`` (a retained
        witness, or a neighbouring state's coordinates) is checked first and takes
        the exact fast path -- one forward composition, no SciPy import (#88).
        Otherwise a bounded, deterministic multi-start solve runs, exiting as soon
        as a tolerance-satisfying witness is found. ``"not_found"`` means no witness
        within the budget -- **never** a proof of non-membership, which a finite set
        of local solves cannot establish for a rotation-rich chain (#85).

        Budget (a TOTAL, caller-visible bound, #89):

        * ``max_starts`` -- the maximum number of optimiser starts, counting every
          start (a refining initial guess, the midpoint, the two opposite corners,
          and seeded interior points). ``ChainSolveResult.starts`` reports how many
          actually ran. Default 11.
        * ``max_nfev`` -- a **strict** total cap on smooth-residual objective
          evaluations across all starts, enforced by our own global counter
          (SciPy's ``maxfun`` is only a soft hint under numerical differentiation).
          ``ChainSolveResult.nfev`` reports the total consumed and never exceeds
          ``max_nfev``. Forward compositions in the tolerance/geodesic membership
          checks are not objective evaluations and are not counted. Default 2200.

        Worst case: at most ``max_starts`` starts and exactly ``max_nfev``
        objective evaluations. ``max_starts == 0`` runs no optimiser and returns
        the (un-optimised) midpoint candidate with ``nfev == 0``.

        ``max_starts`` must be a non-boolean integer >= 0; ``max_nfev`` a
        non-boolean integer >= 1; ``tolerance`` a finite positive number -- all
        validated before SciPy is imported. An ``initial_guess`` that is not shaped
        ``(n, 6)`` or is non-finite is ignored; a finite in-shape guess that is out
        of bounds is clipped into bounds and used as a refining start.
        """
        max_starts = self._DEFAULT_MAX_STARTS if max_starts is None else max_starts
        max_nfev = self._DEFAULT_MAX_NFEV if max_nfev is None else max_nfev
        max_starts = _require_int(max_starts, "max_starts", minimum=0)
        max_nfev = _require_int(max_nfev, "max_nfev", minimum=1)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
            raise ValueError(f"tolerance must be a finite positive number, got {tolerance!r}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance!r}")

        n = len(self.TSRs)
        if n == 0:
            return ChainSolveResult("not_found", numpy.empty((0, 6)), float("inf"), 0, 0)

        # A finite, correctly shaped guess is eligible for the fast path and, if it
        # does not validate, as the first refining start; any other guess (wrong
        # shape or non-finite) is ignored (#89).
        guess = None
        if initial_guess is not None:
            try:
                ig = numpy.asarray(initial_guess, dtype=float)
            except (ValueError, TypeError):
                ig = None
            if ig is not None and ig.shape == (n, 6) and numpy.all(numpy.isfinite(ig)):
                guess = ig

        # Fast path: a supplied witness that already validates. One composition via
        # _witness_residual, and crucially no SciPy import (#88).
        if guess is not None:
            residual = self._witness_residual(trans, guess)
            if residual is not None and residual < tolerance:
                return ChainSolveResult("satisfied", guess, residual, 0, 0)

        # Exact path: a single component reduces to the closed-form TSR check.
        if n == 1:
            tsr = self.TSRs[0]
            if tsr.contains(trans):
                coords = numpy.array([tsr.to_xyzrpy(trans)])
                res = geodesic_distance(self.to_transform(coords), trans)
                return ChainSolveResult("satisfied", coords, float(res), 0, 0)
            dist, bw = tsr.distance(trans)
            return ChainSolveResult("not_found", numpy.array([bw]), float(dist), 0, 0)

        # Continuous bounds over all 6*n coordinates. _Bw_cont guarantees
        # lower <= upper even for wrapping rotation intervals; point-bound
        # coordinates are held fixed, never handed to the optimiser -- a zero-width
        # interval has no descent direction and only adds a rank-deficient column.
        lower = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 0] for i in range(n)])
        upper = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 1] for i in range(n)])
        x_full = (lower + upper) / 2.0
        free = upper > lower

        def expand(x_free):
            x = x_full.copy()
            x[free] = x_free
            return x

        def geodesic_at(x_free):
            return geodesic_distance(self.to_transform(expand(x_free).reshape(n, 6)), trans)

        # All-fixed chain: a single candidate pose, no optimiser, no SciPy (#88).
        if not numpy.any(free):
            res = geodesic_at(numpy.empty(0))
            status = "satisfied" if res < tolerance else "not_found"
            return ChainSolveResult(status, x_full.reshape(n, 6), float(res), 0, 0)

        # --- Bounded multi-start cold inverse (#174) ---
        #
        # Projected Levenberg-Marquardt with an analytic Jacobian, in
        # core/_chain_solver.py. It minimises the same scalar the previous L-BFGS-B path
        # did -- the geodesic's arccos rotation term is non-smooth at 0 and breeds local
        # minima, while the chordal term (3 - tr(Rᵀ R')) is smooth and zero iff the
        # rotations match -- but that scalar is exactly ||r||^2 for a 12-vector, so the
        # problem is least squares and its Jacobian is closed-form. No SciPy here, and
        # no finite differences: a gradient costs no extra chain composition where it
        # used to cost 6n.
        #
        # `nfev` still counts smooth-residual evaluations, one per forward composition,
        # and is still a strict TOTAL cap enforced by our own counter (#91). The forward
        # compositions in the tolerance/geodesic checks are membership tests, not
        # objective calls, and are not counted.
        lo_f, hi_f = lower[free], upper[free]
        span_f = hi_f - lo_f
        n_free = int(free.sum())

        # Deterministic start schedule in priority order, truncated to max_starts
        # total (#89): a supplied-but-invalid guess refines first (canonicalised
        # into the continuous chart via the SAME helper as to_transform, so a valid
        # wrapping coordinate is preserved, not clipped to a boundary, #90), then
        # midpoint, the two opposite corners, then a fixed low-discrepancy set.
        # Unchanged by #174, so the schedule a given query starts from is the same.
        schedule = []
        if guess is not None:
            schedule.append(self._to_continuous(guess).reshape(-1)[free])
        schedule += [x_full[free], lo_f, hi_f]
        gen = numpy.random.default_rng(0)
        while len(schedule) < max_starts:
            schedule.append(lo_f + span_f * gen.random(n_free))
        schedule = schedule[:max_starts]

        # An empty schedule (max_starts == 0) runs no optimiser and leaves the midpoint in
        # the best-point slot, which is what makes max_starts=0 well-defined.
        best_x, nfev, starts = _chain_solver.minimise(
            self,
            trans,
            x_full=x_full,
            lower=lower,
            upper=upper,
            free=free,
            schedule=schedule,
            max_nfev=max_nfev,
            tolerance=tolerance,
            geodesic_at=geodesic_at,
        )

        geo = geodesic_at(best_x)
        status = "satisfied" if geo < tolerance else "not_found"
        return ChainSolveResult(status, expand(best_x).reshape(n, 6), float(geo), nfev, starts)

    def distance(self, trans):
        """
        Best-found geodesic residual from the chain to a transform, and the
        recovered coordinates.

        Delegates to :meth:`solve`. If a witness within tolerance is found the
        residual is ~0; otherwise it is the **best found** residual -- an upper
        bound on the true minimum, not a certified distance (#85). Callers needing
        an exact positive certificate should use :meth:`validate_witness`.

        @param trans 4x4 transform
        @return dist  best-found geodesic residual
        @return bwopt recovered coordinates as an (n, 6) array of xyzrpy
        """
        result = self.solve(trans)
        return result.residual, result.coordinates

    def closest_transform(self, trans):
        """
        Best-found residual and the closest composed world-frame transform.

        Mirrors :meth:`TSR.closest_transform` for chains (#63); inherits
        :meth:`distance`'s best-found (upper-bound) approximation status (#85).
        """
        dist, bwopt = self.distance(trans)
        return dist, self.to_transform(bwopt)

    def contains(self, trans, initial_guess=None):
        """
        Whether the chain contains the transform.

        A chain is the set of poses reachable by **serially composing** a
        transform sampled from each component TSR -- not the Boolean intersection
        of the components' world-frame pose sets. A single-TSR chain uses the
        exact closed-form check. For multi-TSR chains this delegates to
        :meth:`solve`: ``True`` means a witness was found within the numerical
        budget; ``False`` means none was found and is **not** a certificate of
        non-membership (#85). Pass ``initial_guess`` (e.g. a retained witness) for
        the exact fast path.

        @param  trans 4x4 transform
        @param  initial_guess (optional) (n, 6) coordinates to check first
        @return       True if a witness is found, False otherwise
        """
        if len(self.TSRs) == 0:
            return False
        if len(self.TSRs) == 1:
            return self.TSRs[0].contains(trans)
        return self.solve(trans, initial_guess=initial_guess).status == "satisfied"

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to a list of xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy_list list of xyzrpy values
        """
        _, xyzrpy_array = self.distance(trans)
        # Convert numpy array to list of arrays
        return [xyzrpy_array[i] for i in range(len(self.TSRs))]

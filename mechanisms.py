import math
import numbers
from dataclasses import dataclass

import torch
try:
    from scipy.special import erf
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight environments
    from math import erf


def _to_scalar(value, name):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f'{name} must be a scalar, got tensor with shape {tuple(value.shape)}.')
        return float(value.item())
    return float(value)


def _parse_norm_scale(norm_scale):
    if norm_scale is None:
        return None

    if isinstance(norm_scale, bool):
        raise ValueError(
            f'norm_scale must be "none" or a finite positive float, got {norm_scale}.'
        )

    candidate = norm_scale
    if isinstance(norm_scale, str):
        candidate = norm_scale.strip()
        if candidate.lower() == 'none':
            return None

    try:
        value = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'norm_scale must be "none" or a finite positive float, got {norm_scale}.'
        ) from exc

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f'norm_scale must be "none" or a finite positive float, got {norm_scale}.'
        )
    return value


def _normalize_m_mode(m):
    if isinstance(m, str):
        value = m.strip().lower()
        if value in {'best', 'max'}:
            return value
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported m value: {m}. Use 'best', 'max', or a positive integer."
            ) from exc

    if isinstance(m, bool):
        raise ValueError(
            f"Unsupported m value: {m}. Use 'best', 'max', or a positive integer."
        )

    if isinstance(m, numbers.Integral):
        return int(m)

    if isinstance(m, numbers.Real) and float(m).is_integer():
        return int(m)

    raise ValueError(
        f"Unsupported m value: {m}. Use 'best', 'max', or a positive integer."
    )


def _resolve_sample_dimensions(d, eps, m, best_divisor, mechanism_name):
    if d < 1:
        raise ValueError(f'{mechanism_name} requires feature dimension >= 1, got {d}.')

    m_mode = _normalize_m_mode(m)
    if m_mode == 'best':
        if eps < 0:
            raise ValueError(f'eps must be >= 0 for {mechanism_name}.')
        return int(max(1, min(d, math.floor(eps / best_divisor))))
    if m_mode == 'max':
        return d

    m_int = int(m_mode)
    if m_int < 1:
        raise ValueError(f'm must be >= 1 for {mechanism_name}, got {m_int}.')
    if m_int > d:
        raise ValueError(
            f'm must be <= feature dimension ({d}) for {mechanism_name}, got {m_int}.'
        )
    return m_int


def _sample_coordinate_mask(x_unit, m):
    sampled_indices = torch.rand_like(x_unit).topk(m, dim=1).indices
    return torch.zeros_like(x_unit, dtype=torch.bool).scatter(1, sampled_indices, True)


def _multibit_kappa(eps_per_dim):
    # kappa = (exp(eps)-1)/(exp(eps)+1) = tanh(eps/2), which avoids overflow at large eps.
    return math.tanh(eps_per_dim / 2.0)


def _piecewise_stable_terms(eps_per_dim):
    half_eps = eps_per_dim / 2.0
    q = math.exp(-half_eps)
    one_minus_q = -math.expm1(-half_eps)
    one_plus_q = 1.0 + q

    if one_minus_q <= 0.0 or not math.isfinite(one_minus_q):
        raise ValueError('eps is too small for PM to compute a finite unbiased estimator.')

    # Stable re-parameterization using q = exp(-eps/2):
    #   kappa = (1-q)/(1+q), c = (1+q)/(1-q), c-1 = 2q/(1-q)
    kappa = one_minus_q / one_plus_q
    c = one_plus_q / one_minus_q
    c_minus_one = (2.0 * q) / one_minus_q

    # Probabilistic mass terms:
    #   threshold_left = q(x+1)/(2(1+q))
    #   threshold_right = threshold_left + 1/(1+q)
    left_mass_coeff = q / (2.0 * one_plus_q)
    middle_mass = 1.0 / one_plus_q

    return kappa, c, c_minus_one, left_mass_coeff, middle_mass


@dataclass
class PerturbationState:
    z: torch.Tensor
    d: int
    m: int
    eps_per_dim: float
    kappa: float
    effective_gain: float
    unit_support_scale: float



class Mechanism:
    def __init__(self, eps, input_range, norm=False, **kwargs):
        self.eps = eps
        self.norm = bool(norm)
        self.output_range = None
        alpha, beta = input_range
        self.alpha = _to_scalar(alpha, 'input_range lower bound')
        self.beta = _to_scalar(beta, 'input_range upper bound')
        if self.beta <= self.alpha:
            raise ValueError(f'Invalid input_range ({self.alpha}, {self.beta}): require lower < upper.')

    def __call__(self, x):
        raise NotImplementedError

    @property
    def domain_length(self):
        return self.beta - self.alpha

    def to_unit_interval(self, x):
        return (x - self.alpha) * 2.0 / self.domain_length - 1.0

    def restore_scale(self, x):
        return (x + 1.0) * self.domain_length / 2.0 + self.alpha


class FourStageMechanism(Mechanism):
    mechanism_name = 'mechanism'

    def __init__(self, *args, norm_scale='none', **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_scale = _parse_norm_scale(norm_scale)
        self.last_m = None
        self.last_kappa = None
        self.last_c = None
        self.last_effective_gain = None
        self.last_eps_per_dim = None
        self.last_unit_support_scale = None

    def perturb(self, x_unit):
        raise NotImplementedError

    def rectify(self, state):
        if not math.isfinite(state.effective_gain) or state.effective_gain <= 0:
            raise ValueError(f'{self.mechanism_name} unbiased estimator is undefined when eps/m is non-finite or too small. ' 'Use a larger eps > 0.')
        return state.z / state.effective_gain

    def _update_diagnostics(self, state):
        self.last_m = int(state.m)
        self.last_kappa = float(state.kappa)
        self.last_effective_gain = float(state.effective_gain)
        self.last_eps_per_dim = float(state.eps_per_dim)
        self.last_unit_support_scale = float(state.unit_support_scale)
        if math.isfinite(state.effective_gain) and state.effective_gain > 0:
            self.last_c = float((self.domain_length / 2.0) / state.effective_gain)
        else:
            self.last_c = None

    def normalize_output(self, noisy_x, state):
        if self.norm_scale is not None:
            return noisy_x * (1.0 / self.norm_scale)

        if not math.isfinite(state.effective_gain) or state.effective_gain <= 0:
            raise ValueError(
                f'{self.mechanism_name} normalization is undefined when effective_gain is non-finite or too small.'
            )
        return noisy_x * state.effective_gain

    def compute_output_range(self, state):
        gain = float(state.effective_gain)
        support = float(state.unit_support_scale)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError(
                f'{self.mechanism_name} output_range is undefined when effective_gain is non-finite or too small.'
            )

        base_output_range = support / gain
        if not self.norm:
            return float(base_output_range)
        if self.norm_scale is None:
            return float(support)
        return float(base_output_range / self.norm_scale)

    def __call__(self, x):
        x_unit = self.to_unit_interval(x)               # Map inputs to [-1, 1].
        state = self.perturb(x_unit)                    # Perturb using the selected mechanism: MBM -> {-1, 0, 1}, HDS -> [-1-b, 1+b], PM -> [-1, 1].

        
        self._update_diagnostics(state)             
        x_unit_hat = self.rectify(state)                # Correct the estimator's bias.
        # noisy_x = self.restore_scale(x_unit_hat)      # Leave disabled to keep unsampled dimensions at zero.
        output = x_unit_hat
        if self.norm:
            output = self.normalize_output(x_unit_hat,state)    # Normalize the output.
        self.output_range = self.compute_output_range(state)
        return output


class SampledFourStageMechanism(FourStageMechanism):
    best_divisor = None

    def __init__(self, *args, m='best', **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def _resolve_m(self, d):
        return _resolve_sample_dimensions(
            d=d,
            eps=self.eps,
            m=self.m,
            best_divisor=self.best_divisor,
            mechanism_name=self.mechanism_name,
        )


class MultiBit(SampledFourStageMechanism):
    mechanism_name = 'MBM/1BM'
    best_divisor = 2.18


    def _perturb_multibit_unit(self,x_unit, eps_per_dim):
        if eps_per_dim < 0:
            raise ValueError('eps must be >= 0 for MBM/1BM.')

        kappa = _multibit_kappa(eps_per_dim)
        p = (1.0 + kappa * x_unit) / 2.0
        p = torch.clamp(p, min=0.0, max=1.0)
        y = 2.0 * torch.bernoulli(p) - 1.0
        return y, kappa, 1.0

    def perturb(self, x_unit):
        _, d = x_unit.size()
        m = self._resolve_m(d)
        eps_per_dim = self.eps / m
        s = _sample_coordinate_mask(x_unit, m)
        y, kappa, unit_support_scale = self._perturb_multibit_unit(x_unit, eps_per_dim)
        effective_gain = (m / d) * kappa
        return PerturbationState(
            z=s * y,
            d=d,
            m=m,
            eps_per_dim=eps_per_dim,
            kappa=kappa,
            effective_gain=effective_gain,
            unit_support_scale=unit_support_scale,
        )


class OneBit(MultiBit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, m='max', **kwargs)





class MultiDimPiecewise(SampledFourStageMechanism):
    mechanism_name = 'PM'
    best_divisor = 2.5


    def _perturb_piecewise_unit(self, x_unit, eps_per_dim):
        if eps_per_dim <= 0:
            raise ValueError('eps must be > 0 for PM.')

        kappa, c, c_minus_one, left_mass_coeff, middle_mass = _piecewise_stable_terms(eps_per_dim)

        # Equivalent to the original left/right bounds, but avoids c-1 cancellation when eps is huge.
        left = x_unit - 0.5 * (1.0 - x_unit) * c_minus_one
        right = x_unit + 0.5 * (1.0 + x_unit) * c_minus_one

        threshold_left = left_mass_coeff * (x_unit + 1.0)
        threshold_right = threshold_left + middle_mass
        threshold_left = torch.clamp(threshold_left, min=0.0, max=1.0)
        threshold_right = torch.clamp(threshold_right, min=0.0, max=1.0)

        t = torch.rand_like(x_unit)
        mask_left = t < threshold_left
        mask_middle = (t >= threshold_left) & (t <= threshold_right)
        mask_right = t > threshold_right

        y_raw = mask_left * (torch.rand_like(x_unit) * (left + c) - c)
        y_raw += mask_middle * (torch.rand_like(x_unit) * (right - left) + left)
        y_raw += mask_right * (torch.rand_like(x_unit) * (c - right) + right)
        y = y_raw / c
        return y, kappa, 1.0

    def perturb(self, x_unit):
        _, d = x_unit.size()
        m = self._resolve_m(d)
        eps_per_dim = self.eps / m
        s = _sample_coordinate_mask(x_unit, m)
        y, kappa, unit_support_scale = self._perturb_piecewise_unit(x_unit, eps_per_dim)
        effective_gain = (m / d) * kappa
        return PerturbationState(
            z=s * y,
            d=d,
            m=m,
            eps_per_dim=eps_per_dim,
            kappa=kappa,
            effective_gain=effective_gain,
            unit_support_scale=unit_support_scale,
        )


class HighDimSquareWave(SampledFourStageMechanism):
    mechanism_name = 'HDS'
    best_divisor = 3.317



    def _perturb_squarewave_unit(self, x_unit, eps_per_dim):
        if eps_per_dim <= 0:
            raise ValueError('eps must be > 0 for HDS.')

        q = math.exp(-eps_per_dim)
        one_minus_q = -math.expm1(-eps_per_dim)
        numerator_b = q * (eps_per_dim - one_minus_q)
        denominator_b = one_minus_q - eps_per_dim * q
        b = numerator_b / denominator_b
        kappa = 1.0 - one_minus_q / eps_per_dim

        left_mass_coeff = denominator_b / (2.0 * eps_per_dim * one_minus_q)
        middle_mass = (eps_per_dim - one_minus_q) / (eps_per_dim * one_minus_q)

        threshold_left = left_mass_coeff * (x_unit + 1.0)
        threshold_right = threshold_left + middle_mass

        t = torch.rand_like(x_unit)
        mask_left = t < threshold_left
        mask_middle = (t >= threshold_left) & (t <= threshold_right)
        mask_right = t > threshold_right

        y = mask_left * (torch.rand_like(x_unit) * (x_unit + 1.0) - b - 1.0)
        y += mask_middle * (torch.rand_like(x_unit) * 2.0 * b + (x_unit - b))
        y += mask_right * (torch.rand_like(x_unit) * (1.0 - x_unit) + (x_unit + b))
        return y, kappa, 1.0 + b

    def perturb(self, x_unit):
        _, d = x_unit.size()
        m = self._resolve_m(d)
        eps_per_dim = self.eps / m
        s = _sample_coordinate_mask(x_unit, m)
        y, kappa, unit_support_scale = self._perturb_squarewave_unit(x_unit, eps_per_dim)
        effective_gain = (m / d) * kappa
        return PerturbationState(
            z=s * y,
            d=d,
            m=m,
            eps_per_dim=eps_per_dim,
            kappa=kappa,
            effective_gain=effective_gain,
            unit_support_scale=unit_support_scale,
        )






class AdditiveNoiseMechanism(Mechanism):
    def generate_noise(self, x):
        raise NotImplementedError

    def add_noise(self, x, noise):
        return x + noise

    def normalize_output(self, noisy_x):
        max_abs = noisy_x.abs().max()
        if not torch.isfinite(max_abs):
            raise ValueError('Feature normalization failed: found non-finite max_abs.')

        max_abs_value = float(max_abs.item())
        if max_abs_value == 0.0:
            return torch.zeros_like(noisy_x)
        return noisy_x / max_abs_value

    def __call__(self, x):
        noise = self.generate_noise(x)
        noisy_x = self.add_noise(x, noise)
        if self.norm:
            return self.normalize_output(noisy_x)
        return noisy_x


class Laplace(AdditiveNoiseMechanism):
    def _generate_laplace_noise(self, x, scale):
        scale_tensor = torch.ones_like(x) * scale
        return torch.distributions.Laplace(torch.zeros_like(x), scale_tensor).sample()

    def generate_noise(self, x):
        d = x.size(1)
        sensitivity = self.domain_length * d
        scale = sensitivity / self.eps
        out = self._generate_laplace_noise(x, scale)
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out


class Gaussian(AdditiveNoiseMechanism):
    def __init__(self, *args, delta=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.sigma = None
        self.sensitivity = None

    def _generate_gaussian_noise(self, x, sigma):
        std = torch.ones_like(x) * sigma
        return torch.normal(mean=torch.zeros_like(x), std=std)

    def generate_noise(self, x):
        len_interval = self.beta - self.alpha
        if torch.is_tensor(len_interval) and len(len_interval) > 1:
            self.sensitivity = torch.norm(len_interval, p=2)
        else:
            d = x.size(1)
            self.sensitivity = len_interval * math.sqrt(d)

        self.sigma = self.calibrate_gaussian_mechanism()
        out = self._generate_gaussian_noise(x, self.sigma)
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out

    def calibrate_gaussian_mechanism(self):
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps


class AnalyticGaussian(AdditiveNoiseMechanism):
    def __init__(self, *args, delta=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.sigma = None
        self.sensitivity = None

    def _generate_analytic_gaussian_noise(self, x, sigma):
        std = torch.ones_like(x) * sigma
        return torch.normal(mean=torch.zeros_like(x), std=std)

    def generate_noise(self, x):
        len_interval = self.beta - self.alpha
        if torch.is_tensor(len_interval) and len(len_interval) > 1:
            self.sensitivity = torch.norm(len_interval, p=2)
        else:
            d = x.size(1)
            self.sensitivity = len_interval * math.sqrt(d)

        self.sigma = self.calibrate_gaussian_mechanism()
        out = self._generate_analytic_gaussian_noise(x, self.sigma)
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out

    def calibrate_gaussian_mechanism(self, tol=1.e-12):
        """ Calibrate a Gaussian perturbation for differential privacy
        using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
        Arguments:
        tol : error tolerance for binary search (tol > 0)
        Output:
        sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
        """
        delta_thr = self._case_a(0.0)
        if self.delta == delta_thr:
            alpha = 1.0
        else:
            if self.delta > delta_thr:
                predicate_stop_DT = lambda s: self._case_a(s) >= self.delta
                function_s_to_delta = lambda s: self._case_a(s)
                predicate_left_BS = lambda s: function_s_to_delta(s) > self.delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
            else:
                predicate_stop_DT = lambda s: self._case_b(s) <= self.delta
                function_s_to_delta = lambda s: self._case_b(s)
                predicate_left_BS = lambda s: function_s_to_delta(s) < self.delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)
            predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - self.delta) <= tol
            s_inf, s_sup = self._doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = self._binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)
        sigma = alpha * self.sensitivity / math.sqrt(2.0 * self.eps)
        return sigma

    @staticmethod
    def _phi(t):
        return 0.5 * (1.0 + erf(t / math.sqrt(2.0)))

    def _case_a(self, s):
        return self._phi(math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

    def _case_b(self, s):
        return self._phi(-math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

    @staticmethod
    def _doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    @staticmethod
    def _binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid


supported_feature_mechanisms = {
    'mbm': MultiBit,
    '1bm': OneBit,
    'pm': MultiDimPiecewise,
    'hds': HighDimSquareWave,
    'lpm': Laplace,
    'agm': AnalyticGaussian,
}

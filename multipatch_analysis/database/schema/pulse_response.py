from sqlalchemy.orm import relationship
from . import make_table
from .dataset import PulseResponse, Baseline


__all__ = ['PulseResponseStrength', 'BaselineResponseStrength']


PulseResponseStrength = make_table(
    name='pulse_response_strength',
    comment="Measurements of membrane potential or current deflection following each evoked presynaptic spike.",
    columns=[
        ('pulse_response_id', 'pulse_response.id', '', {'index': True, 'unique': True}),
        
        ('amplitude', 'float', 'The estimated amplitude of the synaptic event (if any) in response to this stimulus.'),
        ('dec_fit_amp', 'float', 'Amplitude of gaussian fit to the deconvolved response.'),
        ('dec_fit_latency', 'float', 'Latency (relative to spike max slope) of gaussian fit to the deconvolved response.'),
        ('dec_fit_sigma', 'float', 'Sigma of gaussian fit to the deconvolved response.'),
        
        ('pos_amp', 'float', 'max-median offset from baseline to pulse response window'),
        ('neg_amp', 'float', 'min-median offset from baseline to pulse response window'),
        ('pos_dec_amp', 'float', 'max-median offset from baseline to pulse response window from devonvolved trace'),
        ('neg_dec_amp', 'float', 'min-median offset from baseline to pulse response window from deconvolved trace'),
        ('pos_dec_latency', 'float', 'duration (seconds) from presynaptic spike max dv/dt until the sample measured in pos_dec_amp'),
        ('neg_dec_latency', 'float', 'duration (seconds) from presynaptic spike max dv/dt until the sample measured in neg_dec_amp'),
        ('crosstalk', 'float', 'trace difference immediately before and after onset of presynaptic stimulus pulse'),
    ]
)

BaselineResponseStrength = make_table(
    name='baseline_response_strength',
    comment="Measurements of membrane potential or current deflection in the absence of presynaptic spikes (provides a measurement of background noise to compare to pulse_response_strength).",
    columns=[
        ('baseline_id', 'baseline.id', '', {'index': True, 'unique': True}),
        ('pos_amp', 'float', 'max-median offset from baseline to pulse response window'),
        ('neg_amp', 'float', 'min-median offset from baseline to pulse response window'),
        ('pos_dec_amp', 'float', 'max-median offset from baseline to pulse response window from devonvolved trace'),
        ('neg_dec_amp', 'float', 'min-median offset from baseline to pulse response window from deconvolved trace'),
        ('pos_dec_latency', 'float', 'duration (seconds) from presynaptic spike max dv/dt until the sample measured in pos_dec_amp'),
        ('neg_dec_latency', 'float', 'duration (seconds) from presynaptic spike max dv/dt until the sample measured in neg_dec_amp'),
        ('crosstalk', 'float', 'trace difference immediately before and after onset of presynaptic stimulus pulse'),
    ]
)

PulseResponse.pulse_response_strength = relationship(PulseResponseStrength, back_populates="pulse_response", cascade="delete", single_parent=True, uselist=False)
PulseResponseStrength.pulse_response = relationship(PulseResponse, back_populates="pulse_response_strength", single_parent=True)

Baseline.baseline_response_strength = relationship(BaselineResponseStrength, back_populates="baseline", cascade="delete", single_parent=True, uselist=False)
BaselineResponseStrength.baseline = relationship(Baseline, back_populates="baseline_response_strength", single_parent=True)
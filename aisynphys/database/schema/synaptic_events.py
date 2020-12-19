from sqlalchemy.orm import relationship
from . import make_table
from .experiment import Cell

__all__ = ['SynapticEvents']


SynapticEvents = make_table(
    name='synaptic_events',
    comment="Analysis of synaptic events detected across all recordings for each cell, including spontaneous and asynchronous events.",
    columns=[
        ('cell_id', 'cell.id', '', {'index': True}),
        ('spontaneous_rate', 'float', 'rate of detected events in quiescent recording regions (no nearby stimuli on any channel)'),
        ('events', 'object', 'a structure describing all detected events'),
    ]
)

Cell.synaptic_events = relationship(SynapticEvents, back_populates="cell", cascade='save-update,merge,delete', single_parent=True)
SynapticEvents.cell = relationship(Cell, back_populates='synaptic_events', single_parent=True)

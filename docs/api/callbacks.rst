Callbacks
=========

.. currentmodule:: minerva_opt.callbacks.ray_callbacks

These callbacks are used internally by the pipelines to report metrics and
checkpoints to Ray Train at the end of each Lightning training epoch.  You
only need to interact with them directly if you want to customise the
checkpointing strategy (see :ref:`the tutorial <checkpointing-strategy>`).


TrainerReportOnIntervalCallback
--------------------------------

.. autoclass:: minerva_opt.callbacks.ray_callbacks.TrainerReportOnIntervalCallback
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


TrainerReportKeepOnlyLastCallback
----------------------------------

.. autoclass:: minerva_opt.callbacks.ray_callbacks.TrainerReportKeepOnlyLastCallback
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

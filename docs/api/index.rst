API Reference
=============

This section documents every public class, method, and function in
``minerva-opt``.

.. toctree::
   :maxdepth: 2

   pipelines
   results
   callbacks

Module overview
---------------

.. autosummary::
   :nosignatures:

   minerva_opt.pipelines.hyperparameter_search.RayHyperParameterSearch
   minerva_opt.pipelines.ablation_study.AblationStudyPipeline
   minerva_opt.results.ablation_results.AblationResults
   minerva_opt.callbacks.ray_callbacks.TrainerReportOnIntervalCallback
   minerva_opt.callbacks.ray_callbacks.TrainerReportKeepOnlyLastCallback

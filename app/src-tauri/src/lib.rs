mod background;
mod commands;
#[cfg(target_os = "linux")]
pub(crate) mod host_env;

use commands::{
    algorithms, arrow, cloud, corpus, data, hpo, mlflow, policies, process, sim_watcher, system,
    telemetry, tensor, zenml,
};

#[cfg(desktop)]
use std::sync::Mutex;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "linux")]
    host_env::prepare();

    let builder = tauri::Builder::default();
    // Launching the Studio again shows the running (possibly hidden) window; must be the first plugin.
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
        background::show_main_window(app);
    }));

    builder
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(background::CloseToBackground::default())
        .on_window_event(background::on_window_event)
        .setup(|app| {
            #[cfg(desktop)]
            {
                let mut updater = tauri_plugin_updater::Builder::new();
                if let Ok(pubkey) = std::env::var("WSMART_UPDATER_PUBKEY") {
                    let trimmed = pubkey.trim();
                    if !trimmed.is_empty() {
                        updater = updater.pubkey(trimmed);
                    }
                }
                app.handle().plugin(updater.build())?;
                app.manage(system::PendingUpdate(Mutex::new(None)));
                background::setup_tray(app.handle());
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Data loading
            arrow::csv_to_arrow_ipc,
            arrow::simulation_log_to_arrow_ipc,
            arrow::benchmark_arrow_pipeline,
            arrow::read_binary_file,
            arrow::path_exists,
            data::load_simulation_log,
            corpus::scan_simulation_corpus,
            data::scan_simulation_directory,
            data::load_policy_viz_log,
            data::load_sim_failure_log,
            data::load_training_health_log,
            data::load_attention_viz_log,
            telemetry::load_policy_telemetry_trends,
            telemetry::load_policy_trajectory_trends,
            data::load_csv_file,
            data::list_output_dirs,
            data::list_training_runs,
            data::load_training_metrics,
            data::read_text_file,
            data::write_text_file,
            data::write_binary_file,
            data::list_files_recursive,
            data::list_dir,
            data::preview_dataset_stats,
            data::inspect_wsroute_bundle,
            data::create_wsroute_bundle,
            data::extract_wsroute_bundle,
            data::export_csv_to_parquet,
            data::export_table_parquet,
            // TensorDict / NumPy archives (§G.5.1)
            tensor::inspect_npz_archive,
            tensor::load_tensor_slice,
            tensor::load_npz_vectors,
            tensor::load_npz_flat,
            tensor::tensor_slice_to_arrow_ipc,
            tensor::probe_npy_mmap,
            // Simulation file watcher
            sim_watcher::start_sim_watcher,
            sim_watcher::stop_sim_watcher,
            // Process management
            process::spawn_python_process,
            process::cancel_process,
            process::list_processes,
            // Policy registry
            policies::list_sim_policies,
            algorithms::list_policy_modules,
            algorithms::trace_policy,
            // Optuna HPO
            hpo::list_optuna_studies,
            hpo::load_optuna_study,
            hpo::export_optuna_reports,
            // MLflow experiment tracking
            mlflow::list_mlflow_runs,
            mlflow::list_mlflow_metric_keys,
            mlflow::load_mlflow_metric_history,
            // ZenML pipeline tracking
            zenml::list_zenml_pipeline_runs,
            zenml::load_zenml_run_steps,
            // System inspection
            system::validate_project_root,
            system::probe_python,
            system::dump_hydra_config,
            system::get_app_version,
            system::check_for_updates,
            system::install_app_update,
            // Background mode: closing the window keeps the Studio running
            background::set_close_to_background,
            // Cloud runner (CR-13)
            cloud::cloud_list_jobs,
            cloud::cloud_job_status,
            cloud::cloud_submit,
            cloud::cloud_cost_estimate,
            cloud::cloud_cancel,
            cloud::cloud_retry,
            cloud::cloud_task_logs,
            cloud::cloud_task_metrics,
            cloud::cloud_download,
            cloud::cloud_verify_download,
            cloud::cloud_backends,
            cloud::cloud_worker_tick,
            cloud::cloud_store_credential,
            cloud::cloud_delete_credential,
            cloud::cloud_has_credential,
            cloud::cloud_watch_job,
            cloud::cloud_unwatch_job,
            cloud::cloud_stop_all_watchers,
            cloud::cloud_list_watched_jobs,
        ])
        .run(tauri::generate_context!())
        .expect("error while running WSmart-Route Studio");
}

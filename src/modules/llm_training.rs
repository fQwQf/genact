//! Pretend to train an LLM
use async_trait::async_trait;
use humansize::{FormatSizeOptions, format_size};
use instant::Instant;
use rand::RngExt;
use rand::rng;
use rand::seq::IndexedRandom;

use crate::args::AppConfig;
use crate::data::{LLM_GPUS_LIST, LLM_MODELS_LIST};
use crate::io::{csleep, erase_line, newline, print};
use crate::modules::Module;

pub struct Gpu<'a> {
    pub name: &'a str,
    pub vram: u64,
}

pub struct LlmTraining;

/// Formats a time duration in seconds as MM:SS
///
/// Example: 345 seconds -> "05:45"
fn format_secs_as_mm_ss(secs: u64) -> String {
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

/// Advances work at `units_per_sec`, waiting through `csleep` between updates.
/// `render` receives (step, pct, elapsed_secs, remaining_secs, bar).
/// Elapsed time is measured; ETA is unknown until some work has completed.
async fn run_progress_bar<F>(total: usize, units_per_sec: f64, mut render: F, appconfig: &AppConfig)
where
    F: FnMut(usize, f64, f64, f64, &progress_string::Bar) -> String,
{
    let mut bar = progress_string::BarBuilder::new()
        .total(total)
        .full_char('█')
        .width(10)
        .build();

    let started = Instant::now();
    let mut completed = 0.0;

    loop {
        if appconfig.should_exit() {
            return;
        }

        let step = completed as usize;
        let pct = step as f64 / total as f64 * 100.0;
        let elapsed_secs = started.elapsed().as_secs_f64();
        let remaining_secs = if step > 0 {
            elapsed_secs * (total - step) as f64 / step as f64
        } else {
            0.0
        };

        bar.replace(step);
        erase_line().await;
        print(render(step, pct, elapsed_secs, remaining_secs, &bar)).await;

        if step == total {
            return;
        }

        // Like the download module, advance by throughput times a 50 ms tick.
        // Retain fractional units so slow progress is not rounded away each tick.
        let next = (completed + units_per_sec * 0.05).min(total as f64);
        // Shorten the final tick for files or jobs smaller than one full tick.
        let tick_secs = ((total as f64 - completed) / units_per_sec).min(0.05);
        let sleep_ms = (tick_secs * 1000.0).ceil() as u64;
        csleep(sleep_ms.max(1)).await;
        completed = next;
    }
}

#[async_trait(?Send)]
impl Module for LlmTraining {
    fn name(&self) -> &'static str {
        "llm_training"
    }

    fn signature(&self) -> String {
        "python train.py --model gpt2 --dataset wikitext".to_string()
    }

    async fn run(&self, appconfig: &AppConfig) {
        let mut rng = rng();

        // selects a random GPU and logs the device details
        let gpu = LLM_GPUS_LIST.choose(&mut rng).unwrap();
        // Use format_size to properly display VRAM in human-readable units (e.g., "80 GB")
        let vram_bytes = gpu.vram * 1024 * 1024; // MB -> bytes
        let vram_opts = FormatSizeOptions::from(humansize::BINARY).space_after_value(false);
        print(format!(
            "[INFO] Device found: {} ({})",
            gpu.name,
            format_size(vram_bytes, vram_opts)
        ))
        .await;
        newline().await;
        csleep(300).await;

        // chooses mixed precision training using either "bf16" or "fp16"
        let precision = if rng.random_bool(0.6) { "bf16" } else { "fp16" };
        print(format!(
            "[INFO] Using mixed precision training ({precision})."
        ))
        .await;
        newline().await;

        // configures gradient accumulation steps, micro-batch size, and calculates the total batch size
        let grad_acc_steps = [2u32, 4, 8, 16].choose(&mut rng).unwrap();
        let micro_batch = [8u32, 16, 32].choose(&mut rng).unwrap();
        let total_batch = grad_acc_steps * micro_batch;
        print(format!(
            "[INFO] Gradient accumulation steps: {grad_acc_steps} | Total batch size: {total_batch}"
        ))
        .await;
        newline().await;

        // initializes weights from a randomly selected model
        let model = LLM_MODELS_LIST.choose(&mut rng).unwrap();
        print(format!(
            "[INFO] Initializing model weights from '{model}'... Done."
        ))
        .await;
        newline().await;
        csleep(500).await;

        // downloads model files
        let file_sizes: Vec<u64> = vec![
            rng.random_range(400_000..3_000_000),
            rng.random_range(10_000..80_000),
            rng.random_range(200_000_000..7_000_000_000),
            rng.random_range(1_000..8_000),
        ];

        // Share a connection speed across files, so larger files take longer.
        let download_bytes_per_sec = rng.random_range(10_000_000u64..100_000_000);

        for &file_size in &file_sizes {
            // Track progress in KiB so file sizes fit in wasm32's usize.
            let file_size_kib = file_size.div_ceil(1024);
            let file_size = file_size_kib * 1024;

            let size_opts = FormatSizeOptions::from(humansize::BINARY).space_after_value(false);
            let speed_opts = FormatSizeOptions::from(humansize::BINARY)
                .space_after_value(false)
                .suffix("/s");

            run_progress_bar(
                file_size_kib as usize,
                download_bytes_per_sec as f64 / 1024.0,
                move |downloaded, pct, elapsed_secs, remaining_secs, bar| {
                    if downloaded == 0 || elapsed_secs <= 0.0 {
                        format!(
                            "{pct:>3.0}%{bar} {downloaded}/{total} [{elapsed}<?, ?/s]",
                            downloaded = format_size(downloaded as u64 * 1024, size_opts),
                            total = format_size(file_size, size_opts),
                            elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                        )
                    } else {
                        format!(
                            "{pct:>3.0}%{bar} {downloaded}/{total} [{elapsed}<{eta}, {speed}]",
                            downloaded = format_size(downloaded as u64 * 1024, size_opts),
                            total = format_size(file_size, size_opts),
                            elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                            eta = format_secs_as_mm_ss(remaining_secs.ceil() as u64),
                            speed = format_size(
                                (downloaded as f64 * 1024.0 / elapsed_secs) as u64,
                                speed_opts,
                            ),
                        )
                    }
                },
                appconfig,
            )
            .await;

            if appconfig.should_exit() {
                newline().await;
                return;
            }

            newline().await;
        }

        // logs the number of instances in the training dataset
        let num_instances = rng.random_range(10_000..100_000);
        print(format!("Training set has {num_instances} instances")).await;
        newline().await;
        newline().await;

        let num_epochs = rng.random_range(3..8);
        let steps_per_epoch = rng.random_range(500..2000);
        let mut loss = rng.random_range(3.0..5.0);

        // Keep training throughput consistent across epochs of the same run.
        let training_steps_per_sec = rng.random_range(10.0..40.0);

        // do the training process over a random number of epochs with progress bars
        for epoch in 1..=num_epochs {
            run_progress_bar(
                steps_per_epoch as usize,
                training_steps_per_sec,
                |step, pct, elapsed_secs, remaining_secs, bar| {
                    if step == 0 || elapsed_secs <= 0.0 {
                        format!(
                            "Epoch {epoch}/{num_epochs}: {pct:>3.0}%{bar} {step}/{steps_per_epoch} [{elapsed}<?, ?it/s]",
                            elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                        )
                    } else {
                        let it_per_s = step as f64 / elapsed_secs;
                        let step_loss = (loss
                            - rng.random_range(0.0..0.2) * step as f64 / steps_per_epoch as f64)
                            .max(0.05);
                        format!(
                            "Epoch {epoch}/{num_epochs}: {pct:>3.0}%{bar} {step}/{steps_per_epoch} [{elapsed}<{eta}, {it:.2}it/s, loss={loss:.4}]",
                            elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                            eta = format_secs_as_mm_ss(remaining_secs.ceil() as u64),
                            it = it_per_s,
                            loss = step_loss,
                        )
                    }
                },
                appconfig,
            )
            .await;

            if appconfig.should_exit() {
                newline().await;
                return;
            }

            let prev_loss = loss;
            loss -= rng.random_range(0.3..0.8);
            loss = loss.max(0.05);
            newline().await;

            // Occasionally save a checkpoint when the loss improves.
            if loss < prev_loss && rng.random_bool(0.4) {
                let checkpoint_step = epoch * steps_per_epoch;
                print(format!(
                    "[CHECKPOINT] Model improved ({prev_loss:.4} -> {loss:.4}). Saving model to ./output/checkpoint-{checkpoint_step}..."
                ))
                .await;
                newline().await;
                csleep(rng.random_range(1_000..2_000)).await;
                print(format!(
                    "[INFO] Configuration saved in ./output/checkpoint-{checkpoint_step}/config.json"
                ))
                .await;
                newline().await;
            }

            if appconfig.should_exit() {
                return;
            }
        }

        let val_steps = rng.random_range(50..300);
        let validation_steps_per_sec = rng.random_range(30.0..100.0);

        // we do a validation step after training
        run_progress_bar(
            val_steps as usize,
            validation_steps_per_sec,
            |step, pct, elapsed_secs, remaining_secs, bar| {
                if step == 0 || elapsed_secs <= 0.0 {
                    format!(
                        "Validating: {pct:>3.0}%{bar} {step}/{val_steps} [{elapsed}<?, ?it/s]",
                        elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                    )
                } else {
                    let it_per_s = step as f64 / elapsed_secs;
                    format!(
                        "Validating: {pct:>3.0}%{bar} {step}/{val_steps} [{elapsed}<{eta}, {it:.2}it/s]",
                        elapsed = format_secs_as_mm_ss(elapsed_secs as u64),
                        eta = format_secs_as_mm_ss(remaining_secs.ceil() as u64),
                        it = it_per_s,
                    )
                }
            },
            appconfig,
        )
        .await;

        if appconfig.should_exit() {
            newline().await;
            return;
        }

        newline().await;

        // displays the validation results
        let val_loss = loss + rng.random_range(0.05..0.3);
        let perplexity = val_loss.exp();
        print(format!(
            "> Validation results: loss={val_loss:.4} | perplexity={perplexity:.2}"
        ))
        .await;
        newline().await;

        csleep(500).await;
    }
}

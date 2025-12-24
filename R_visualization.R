######################################################################
# Plot image size and nonzero pixel percentage (manifest-driven EDA)   #
# Revival version: no hardcoded /content paths, no install.packages()  #
######################################################################

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

get_arg <- function(flag, default=NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  if (!(flag %in% args)) return(default)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  return(args[[idx + 1]])
}

stats_csv <- get_arg("--stats-csv", "reports/image_stats.csv")
output_dir <- get_arg("--output-dir", "reports")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(stats_csv)

df_processed <- df %>%
  mutate(
    pathology_processed = case_when(
      pathology == "MALIGNANT" ~ "Malignant",
      pathology == "BENIGN_WITHOUT_CALLBACK" ~ "Benign",
      pathology == "BENIGN" ~ "Benign",
      TRUE ~ as.character(pathology)
    )
  )

########################
# 1) width/height scatter
total_dots <- nrow(df_processed)
formatted_total_dots <- format(total_dots, big.mark = ",")
benign_count <- sum(df_processed$pathology_processed == "Benign", na.rm = TRUE)
malignant_count <- sum(df_processed$pathology_processed == "Malignant", na.rm = TRUE)

full_pixel <- ggplot(df_processed, aes(x = width, y = height, color = pathology_processed)) +
  geom_point(alpha = 0.7) +
  labs(x = "Width (px)", y = "Height (px)", color = "Pathology") +
  scale_color_manual(values = c("Benign" = "#6da9ed", "Malignant" = "#eb6a4d")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(paste(
    "Pixel dimensions for", formatted_total_dots, "images\n",
    "Benign:", benign_count, ", Malignant:", malignant_count
  ))

ggsave(file.path(output_dir, "image_dimensions_scatter.pdf"), full_pixel, width = 7, height = 5.7, dpi = 600)

########################
# 2) nonzero% violin + Wilcoxon

roi_processed <- df_processed %>%
  filter(!is.na(nonzero_percent)) %>%
  filter(pathology_processed %in% c("Benign", "Malignant"))

if (nrow(roi_processed) > 0) {
  wilcox_test_result <- wilcox.test(nonzero_percent ~ pathology_processed, data = roi_processed)
  print(wilcox_test_result$p.value)

  summary_stats <- roi_processed %>%
    group_by(pathology_processed) %>%
    summarise(
      count = n(),
      median = median(nonzero_percent),
      q1 = quantile(nonzero_percent, 0.25),
      q3 = quantile(nonzero_percent, 0.75),
      .groups = "drop"
    )

  total_dots_ROI <- nrow(roi_processed)
  formatted_total_dots_ROI <- format(total_dots_ROI, big.mark = ",")

  malignant_count <- summary_stats$count[summary_stats$pathology_processed == "Malignant"]
  benign_count <- summary_stats$count[summary_stats$pathology_processed == "Benign"]

  ROI_area <- ggplot(roi_processed, aes(x = pathology_processed, y = nonzero_percent, color = pathology_processed)) +
    geom_violin(trim = FALSE) +
    geom_hline(data = summary_stats, aes(yintercept = median, color = pathology_processed), linetype = "dashed", size = 0.5) +
    labs(x = "Pathology", y = "Nonzero pixels (%)", color = "Pathology") +
    scale_color_manual(values = c("Benign" = "#6da9ed", "Malignant" = "#eb6a4d")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    ggtitle(paste(
      "Nonzero pixel % distribution for", formatted_total_dots_ROI, "images\n",
      "Benign:", benign_count, ", Malignant:", malignant_count
    ))

  ggsave(file.path(output_dir, "nonzero_percent_violin.pdf"), ROI_area, width = 6.27, height = 5.7, dpi = 600)
} else {
  message("No usable pathology/nonzero_percent values found; skipping violin plot.")
}

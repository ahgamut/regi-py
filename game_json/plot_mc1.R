library(tidyverse)
library(jsonlite)
library(glue)
library(boxr)
box_auth()

source("game_json/gamejson.R")

progress_breaks <- c(0, 20, 40, 60, 80, 
                     110, 140, 170, 200,
                    240, 280, 320, 360)

progress_labels <- c("Start", "1st Jack", "2nd Jack", "3rd Jack", "4th Jack",
                     "1st Queen", "2nd Queen", "3rd Queen", "4th Queen",
                    "1st King", "2nd King", "3rd King", "Win")

# Get parsed JSONs from Box (mc1.zip)
box_dl(file_id = 2157855130959, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
list.files(tempdir())

mc1_file_list <- unzip(glue::glue("{tempdir()}/mc1.zip"), list = TRUE)

mc1_path <- glue("{tempdir()}/mc1.zip")
mc1_csv_path <- "mc1-256.csv"

mc1 <- read_csv(unz(mc1_path, mc1_csv_path), progress = TRUE)

# Find the maximum progress for each game
mc1_max_progress <- mc1 |>
  group_by(game) |>
  summarize(max_progress = max(game.progress, na.rm = TRUE),
            game_length = max(game.phase_count, na.rm = TRUE))

# Find the winning games
win_ids <- mc1_max_progress |>
  filter(max_progress == 360)

mc1_wins <- mc1 |>
  filter(game %in% win_ids$game)

mc1_win1 <- mc1_wins |>
  filter(game == win_ids$game[1])

mc1_win1_list <- split(mc1_win1, seq(nrow(mc1_win1)))

mc1_win1_list |>
  map(\(x) diagram_game_phase(x))

mc1_win2 <- mc1_wins |>
  filter(game == win_ids$game[2])

mc1_win2_list <- split(mc1_win2, seq(nrow(mc1_win2)))

mc1_win2_list |>
  map(\(x) diagram_game_phase(x))
























library(tidyverse)
library(jsonlite)
library(glue)

source("game_json/gamejson.R")

game_jsons <- list.files(path='game_json/small-sub4',pattern="game")
filepaths <- glue("game_json/small-sub4/{game_jsons}")
sim_setup <- str_sub(filepaths, start = 22, end = 43)

parsed_game_jsons <- filepaths |>
  map(possibly(\(x) parse_game_json(x), otherwise = NULL, quiet = FALSE))
names(parsed_game_jsons) <- sim_setup

exgame <- parsed_game_jsons[[1]]

nbad_games <- sum(sapply(parsed_game_jsons, is.null))

good_games <- parsed_game_jsons |>
  compact() # Removes any NULL list elements

good_games_df <- bind_rows(good_games, .id = "id")

good_games_df <- good_games_df |>
  mutate(game = str_sub(id, start = 5, end = 8),
         team = str_sub(id, start = 14, end = 16),
         sim = str_sub(id, start = 21, end = 22)) |>
  select(!id)

enemy_order <- good_games_df |>
  filter(event == "STARTGAME") |>
  select(game, game.enemy_pile) |>
  filter(!is.na(game.enemy_pile)) |>
  slice_head(n = 1, by = game)

team_members <- good_games_df |>
  filter(!is.na(strategy)) |>
  select(team, strategy) |>
  group_by(team, strategy) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(team) |>
  summarize(teammates = str_flatten(strategy, collapse = " & "))

team_mappings <- jsonlite::fromJSON("game_json/small-sub4/mappings.json", flatten = TRUE)$teams

# Summary statistics and plots for progress
progress_df <- good_games_df |>
  group_by(game, team, sim) |>
  summarize(max_progress = max(game.progress, na.rm = TRUE))

progress_summaries_game <- progress_df |>
  group_by(game) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE),
            var_progress = var(max_progress, na.rm = TRUE),
            sd_progress = sd(max_progress, na.rm = TRUE)) |>
  left_join(enemy_order, by = "game")
str_sub(progress_summaries_game$game.enemy_pile, start = 12, end = 12) <- "\n"
str_sub(progress_summaries_game$game.enemy_pile, start = 24, end = 24) <- "\n"

progress_summaries_team <- progress_df |>
  group_by(team) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE),
            var_progress = var(max_progress, na.rm = TRUE),
            sd_progress = sd(max_progress, na.rm = TRUE)) |>
  left_join(team_members, by = "team")

team_strength_plot <- lineplot(df = progress_summaries_team,
                               group = teammates,
                               statistic = mean_progress,
                               xlab = "Average Progress Made",
                               ylab = "Bot Team")

team_strength_plot_top2 <- lineplot(df = progress_summaries_team,
                               group = teammates,
                               statistic = mean_progress,
                               k = 2,
                               xlab = "Average Progress Made",
                               ylab = "Bot Team")

team_strength_plot_bottom2 <- lineplot(df = progress_summaries_team,
                               group = teammates,
                               statistic = mean_progress,
                               k = 2,
                               top = FALSE,
                               xlab = "Average Progress Made",
                               ylab = "Bot Team")

# Which game is easiest?
game_strength_plot <- lineplot(df = progress_summaries_game,
                               group = game.enemy_pile,
                               statistic = mean_progress,
                               xlab = "Average Progress Made",
                               ylab = "Game")

# Summary statistics and plots for phase count
duration_df <- good_games_df |>
  group_by(game, team, sim) |>
  summarize(max_phase_count = max(game.phase_count, na.rm = TRUE))

duration_summaries_game <- duration_df |>
  group_by(game) |>
  summarize(mean_phase_count = mean(max_phase_count, na.rm = TRUE),
            var_phase_count = var(max_phase_count, na.rm = TRUE),
            sd_phase_count = sd(max_phase_count, na.rm = TRUE)) |>
  left_join(enemy_order, by = "game")
str_sub(duration_summaries_game$game.enemy_pile, start = 12, end = 12) <- "\n"
str_sub(duration_summaries_game$game.enemy_pile, start = 24, end = 24) <- "\n"

duration_summaries_team <- duration_df |>
  group_by(team) |>
  summarize(mean_phase_count = mean(max_phase_count, na.rm = TRUE),
            var_phase_count = var(max_phase_count, na.rm = TRUE),
            sd_phase_count = sd(max_phase_count, na.rm = TRUE)) |>
  left_join(team_members, by = "team")

phase_count_summaries <- good_games_df |>
  filter(event == "ENEMYKILL") |>
  group_by(game, team, sim) |>
  mutate(kill_number = row_number()) |>
  left_join(team_members, by = "team")
str_sub(phase_count_summaries$game.enemy_pile, start = 12, end = 12) <- "\n"
str_sub(phase_count_summaries$game.enemy_pile, start = 24, end = 24) <- "\n"

phase_count_boxplots <- phase_count_summaries |>
  ggplot(aes(x = kill_number,
             y = game.phase_count,
            group = kill_number)) +
  geom_boxplot() +
  labs(x = "Enemy Kill Number",
       y = "Phase Number") +
  theme_bw() +
  theme(panel.grid.minor.x = element_blank())

phase_count_boxplots_team <- phase_count_summaries |>
  ggplot(aes(x = kill_number,
             y = game.phase_count,
             group = kill_number)) +
  geom_boxplot() +
  labs(x = "Enemy Kill Number",
       y = "Phase Number") +
  theme_bw() +
  theme(panel.grid.minor.x = element_blank()) +
  facet_wrap(~teammates)

phase_count_boxplots_game <- phase_count_summaries |>
  ggplot(aes(x = kill_number,
             y = game.phase_count,
            group = kill_number)) +
  geom_boxplot() +
  labs(x = "Enemy Kill Number",
       y = "Phase Number") +
  theme_bw() +
  theme(panel.grid.minor.x = element_blank()) +
  facet_wrap(~game)


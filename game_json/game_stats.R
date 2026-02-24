library(tidyverse)
library(jsonlite)
library(glue)
library(boxr)
library(tictoc)

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
            min_progress = min(max_progress, na.rm = TRUE),
            maximum_progress = max(max_progress, na.rm = TRUE)) |>
  left_join(enemy_order, by = "game")
str_sub(progress_summaries_game$game.enemy_pile, start = 12, end = 12) <- "\n"
str_sub(progress_summaries_game$game.enemy_pile, start = 24, end = 24) <- "\n"

progress_summaries_game_plot <- progress_summaries_game |>
  select(game, mean_progress, min_progress, maximum_progress) |>
  mutate(game = reorder(factor(game), desc(mean_progress))) |>
  pivot_longer(cols = c("mean_progress", "min_progress", "maximum_progress"),
               names_to = "statname",
               values_to = "value")

progress_summaries_game_plot_top10 <- progress_summaries_game_plot |>
  filter(game %in% head(levels(game), 10))

ggplot(data = progress_summaries_game_plot_top10,
                            mapping = aes(x = value,
                                          y = game,
                                          group = game,
                                          color = game)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none")+
    expand_limits(x = 0)+
    scale_y_discrete(limits = rev) + 
    scale_color_viridis_d(option = "viridis", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Game")

progress_summaries_game_plot_bottom10 <- progress_summaries_game_plot |>
  filter(game %in% tail(levels(game), 10))

ggplot(data = progress_summaries_game_plot_bottom10,
                            mapping = aes(x = value,
                                          y = game,
                                          group = game,
                                          color = game)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none")+
    expand_limits(x = 0)+
    scale_y_discrete(limits = rev) +
    scale_color_viridis_d(option = "viridis", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Game")

progress_summaries_team <- progress_df |>
  group_by(team) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE)) |>
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

# Poster plots
box_dl(file_id = 2144308291033, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
list.files(tempdir())

file_list <- unzip(glue::glue("{tempdir()}/big-sub4.zip"), list = TRUE)

big_mapping_json <- jsonlite::fromJSON(unz(glue::glue("{tempdir()}/big-sub4.zip"), "mp4/mappings.json"), flatten = TRUE)$teams

file_list_games <- file_list |>
  filter(!(Name %in% c("mp4/", "mp4/mappings.json"))) 

zip_path <- glue::glue("{tempdir()}/big-sub4.zip")

tic()
parsed_game_jsons <- file_list_games$Name |>
  map(possibly(\(x) parse_game_json_zip(zip_path, x), otherwise = NULL, quiet = FALSE))
toc()

sim_setup <- str_sub(file_list_games$Name, start = 5, end = 26)
names(parsed_game_jsons) <- sim_setup

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

# Summary statistics and plots for progress
progress_df <- good_games_df |>
  group_by(game, team, sim) |>
  summarize(max_progress = max(game.progress, na.rm = TRUE))

progress_summaries_team <- progress_df |>
  group_by(team) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE),
            min_progress = min(max_progress, na.rm = TRUE),
            maximum_progress = max(max_progress, na.rm = TRUE)) |>
  left_join(team_members, by = "team")

progress_summaries_team_plot <- progress_summaries_team |>
  select(team, mean_progress, min_progress, maximum_progress, teammates) |>
  mutate(team = reorder(factor(team), desc(mean_progress))) |>
  pivot_longer(cols = c("mean_progress", "min_progress", "maximum_progress"),
               names_to = "statname",
               values_to = "value")

progress_summaries_team_plot_top10 <- progress_summaries_team_plot |>
  filter(team %in% head(levels(team), 10))

ggplot(data = progress_summaries_team_plot_top10,
                            mapping = aes(x = value,
                                          y = team,
                                          group = team,
                                          color = team)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none")+
    expand_limits(x = 0)+
    scale_y_discrete(limits = rev) + 
    scale_color_viridis_d(option = "viridis", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Top 10 Teams")

progress_summaries_team_plot_bottom10 <- progress_summaries_team_plot |>
  filter(team %in% tail(levels(team), 10))

ggplot(data = progress_summaries_team_plot_bottom10,
                            mapping = aes(x = value,
                                          y = team,
                                          group = team,
                                          color = team)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none")+
    expand_limits(x = 0)+
    scale_y_discrete(limits = rev) +
    scale_color_viridis_d(option = "viridis", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Bottom 10 Teams")

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

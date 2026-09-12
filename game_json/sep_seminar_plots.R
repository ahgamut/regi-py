library(tidyverse)
library(jsonlite)
library(glue)
library(boxr)
library(ggridges)
box_auth()

progress_breaks <- c(0, 20, 40, 60, 80, 
                     110, 140, 170, 200,
                    240, 280, 320, 360)

progress_labels <- c("Start", "1st Jack", "2nd Jack", "3rd Jack", "4th Jack",
                     "1st Queen", "2nd Queen", "3rd Queen", "4th Queen",
                    "1st King", "2nd King", "3rd King", "Win")

# Top 10 teams now ----

box_dl(file_id = 2451924659111, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
# list.files(tempdir())

r1_file_list <- unzip(glue::glue("{tempdir()}/2026-09-07-r1-bench-summary.zip"), list = TRUE)

r1_path <- glue("{tempdir()}/2026-09-07-r1-bench-summary.zip")
r1_2p_stats_path <- "2-p/stats.csv"

r1_2p_stats <- read_csv(unz(r1_path, r1_2p_stats_path), progress = TRUE)

r1_2p_enemy_order <- r1_2p_stats |>
  filter(event == "STARTGAME") |>
  select(game, game.enemy_pile) |>
  filter(!is.na(game.enemy_pile)) |>
  slice_head(n = 1, by = game)

r1_2p_team_members <- r1_2p_stats |>
  filter(!is.na(game.active_player.strategy)) |>
  select(team, game.active_player.strategy) |>
  group_by(team, game.active_player.strategy) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(team) |>
  summarize(teammates = str_flatten(game.active_player.strategy, collapse = " & ")) |>
  mutate(teammates = ifelse(str_detect(teammates, "&"), teammates, glue("{teammates} & {teammates}"))
  )

r1_2p_game_progress <- r1_2p_stats |>
  group_by(game, team, sim) |>
  summarize(game_progress = max(game.progress, na.rm = TRUE),
            game_length = max(game.phase_count, na.rm = TRUE)) 

r1_2p_overall_team_progress <- r1_2p_game_progress |> 
  group_by(team) |> 
  summarize(min_progress = min(game_progress, na.rtm = TRUE),
            mean_progress = mean(game_progress, na.rm = TRUE),
            max_progress = max(game_progress, na.rm = TRUE))

# Top 10 teams determined by mean progress across sims
top10_teams <- r1_2p_overall_team_progress |> 
  arrange(desc(mean_progress)) |> 
  slice_head(n = 10) |> 
  pluck("team")

r1_2p_max_progress_top10 <- r1_2p_game_progress |> 
  filter(team %in% top10_teams) |> 
  mutate(team = factor(team, levels = top10_teams))


r1_2p_progress_top10_plot <- ggplot(data = r1_2p_max_progress_top10, 
                             aes(x = game_progress, y = team, group = team, fill = team)) + 
   geom_density_ridges2(show.legend = FALSE) +
   theme_bw() +
   theme(panel.grid.major.y = element_blank(),
         panel.grid.minor.x = element_blank(),
         axis.text.x = element_text(angle = 45, hjust = 1)) +
   scale_x_continuous(breaks = progress_breaks,
                      labels = progress_labels,
                      limits = c(0, 360)) +
   scale_y_discrete(limits = rev) + 
   scale_fill_viridis_d(option = "mako", begin = 0, end = 0.85) +
   labs(x = "Progress", 
        y = "Top 10 Teams",
        title = glue("Game Progress for the Current Top 10 Strategies"))

r1_2p_progress_top10_plot

# How the best brute teams from last time are doing now (AKA number of futures plot for brute) ----
brute_teams <- c("brute-256|brute-256", "brute-128|brute-128",
                 "brute-64|brute-64", "brute-32|brute-32",
                 "brute-16|brute-16")

brutes <- r1_2p_game_progress |> 
  filter(team %in% brute_teams) |> 
  mutate(team = factor(team, levels = brute_teams))

brute_plot <- ggplot(data = brutes, 
                     aes(x = game_progress, y = team, group = team, fill = team)) + 
   geom_density_ridges2(show.legend = FALSE) +
   theme_bw() +
   theme(panel.grid.major.y = element_blank(),
         panel.grid.minor.x = element_blank(),
         axis.text.x = element_text(angle = 45, hjust = 1)) +
   scale_x_continuous(breaks = progress_breaks,
                      labels = progress_labels,
                      limits = c(0, 360)) +
   scale_y_discrete(limits = rev) + 
   scale_fill_viridis_d(option = "magma", begin = 0, end = 0.85) +
   labs(x = "Progress", 
        y = "Brute Team",
        title = glue("Game Progress for the Current Brute Strategies"))

brute_plot

# Number of futures plotadzmulti ----
adzmulti_teams <- c("adz-adzmulti-256|adz-adzmulti-256", "adz-adzmulti-128|adz-adzmulti-128",
              "adz-adzmulti-64|adz-adzmulti-64", "adz-adzmulti-32|adz-adzmulti-32",
            "adz-adzmulti-16|adz-adzmulti-16", "adz-direct-adzmulti|adz-direct-adzmulti")

adzmulti <- r1_2p_game_progress |> 
  filter(team %in% adzmulti_teams) |> 
  mutate(team = factor(team, levels = adzmulti_teams))

adzmulti_plot <- ggplot(data = adzmulti, 
                     aes(x = game_progress, y = team, group = team, fill = team)) + 
   geom_density_ridges2(show.legend = FALSE) +
   theme_bw() +
   theme(panel.grid.major.y = element_blank(),
         panel.grid.minor.x = element_blank(),
         axis.text.x = element_text(angle = 45, hjust = 1)) +
   scale_x_continuous(breaks = progress_breaks,
                      labels = progress_labels,
                      limits = c(0, 360)) +
   scale_y_discrete(limits = rev) + 
   scale_fill_viridis_d(option = "plasma", begin = 0, end = 0.85) +
   labs(x = "Progress", 
        y = "Adzmulti Team",
        title = glue("Game Progress for the Adzmulti Strategies"))

adzmulti_plot

# Info files (bots allowed to cheat) ----
box_dl(file_id = 2461324458761, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
# list.files(tempdir())

r1_info_file_list <- unzip(glue::glue("{tempdir()}/2026-09-08-r1-info-summary.zip"), list = TRUE)

r1_info_path <- glue("{tempdir()}/2026-09-08-r1-info-summary.zip")
r1_info_2p_stats_path <- "2-p/stats.csv"

r1_info_2p_stats <- read_csv(unz(r1_info_path, r1_info_2p_stats_path), progress = TRUE)

r1_info_2p_enemy_order <- r1_info_2p_stats |>
  filter(event == "STARTGAME") |>
  select(game, game.enemy_pile) |>
  filter(!is.na(game.enemy_pile)) |>
  slice_head(n = 1, by = game)

r1_info_2p_team_members <- r1_info_2p_stats |>
  filter(!is.na(game.active_player.strategy)) |>
  select(team, game.active_player.strategy) |>
  group_by(team, game.active_player.strategy) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(team) |>
  summarize(teammates = str_flatten(game.active_player.strategy, collapse = " & ")) |>
  mutate(teammates = ifelse(str_detect(teammates, "&"), teammates, glue("{teammates} & {teammates}"))
  )

r1_info_2p_game_progress <- r1_info_2p_stats |>
  group_by(game, team, sim) |>
  summarize(game_progress = max(game.progress, na.rm = TRUE),
            game_length = max(game.phase_count, na.rm = TRUE)) 

r1_info_2p_overall_team_progress <- r1_info_2p_game_progress |> 
  group_by(team) |> 
  summarize(min_progress = min(game_progress, na.rtm = TRUE),
            mean_progress = mean(game_progress, na.rm = TRUE),
            max_progress = max(game_progress, na.rm = TRUE))

# Top 10 teams determined by mean progress across sims
info_top10_teams <- r1_info_2p_overall_team_progress |> 
  arrange(desc(mean_progress)) |> 
  slice_head(n = 10) |> 
  pluck("team")

info_r1_2p_max_progress_top10 <- r1_info_2p_game_progress |> 
  filter(team %in% info_top10_teams) |> 
  mutate(team = factor(team, levels = info_top10_teams))

info_r1_2p_progress_top10_plot <- ggplot(data = info_r1_2p_max_progress_top10, 
                             aes(x = game_progress, y = team, group = team, fill = team)) + 
   geom_density_ridges2(show.legend = FALSE) +
   theme_bw() +
   theme(panel.grid.major.y = element_blank(),
         panel.grid.minor.x = element_blank(),
         axis.text.x = element_text(angle = 45, hjust = 1)) +
   scale_x_continuous(breaks = progress_breaks,
                      labels = progress_labels,
                      limits = c(0, 360)) +
   scale_y_discrete(limits = rev) + 
   scale_fill_viridis_d(option = "mako", begin = 0, end = 0.85) +
   labs(x = "Progress", 
        y = "Top 10 Teams",
        title = glue("Game Progress for the Current Top 10 Cheating Strategies"))

info_r1_2p_progress_top10_plot
# brute is good at cheating!

# Info2 files (omniscient bots) ----
# no CSVs yet
box_dl(file_id = 2461324382241, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
# list.files(tempdir())

r1_info2_file_list <- unzip(glue::glue("{tempdir()}/2026-09-08-r1-info2-summary.zip"), list = TRUE)

r1_info2_path <- glue("{tempdir()}/2026-09-08-r1-info2-summary.zip")
r1_info2_2p_stats_path <- "2-p/stats.csv"

r1_info2_2p_stats <- read_csv(unz(r1_info2_path, r1_info2_2p_stats_path), progress = TRUE)

r1_info2_2p_enemy_order <- r1_info2_2p_stats |>
  filter(event == "STARTGAME") |>
  select(game, game.enemy_pile) |>
  filter(!is.na(game.enemy_pile)) |>
  slice_head(n = 1, by = game)

r1_info2_2p_team_members <- r1_info2_2p_stats |>
  filter(!is.na(game.active_player.strategy)) |>
  select(team, game.active_player.strategy) |>
  group_by(team, game.active_player.strategy) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(team) |>
  summarize(teammates = str_flatten(game.active_player.strategy, collapse = " & ")) |>
  mutate(teammates = ifelse(str_detect(teammates, "&"), teammates, glue("{teammates} & {teammates}"))
  )

r1_info2_2p_game_progress <- r1_info2_2p_stats |>
  group_by(game, team, sim) |>
  summarize(game_progress = max(game.progress, na.rm = TRUE),
            game_length = max(game.phase_count, na.rm = TRUE)) 

r1_info2_2p_overall_team_progress <- r1_info2_2p_game_progress |> 
  group_by(team) |> 
  summarize(min_progress = min(game_progress, na.rtm = TRUE),
            mean_progress = mean(game_progress, na.rm = TRUE),
            max_progress = max(game_progress, na.rm = TRUE))

# Top 10 teams determined by mean progress across sims
info2_top10_teams <- r1_info2_2p_overall_team_progress |> 
  arrange(desc(mean_progress)) |> 
  slice_head(n = 10) |> 
  pluck("team")

info2_r1_2p_max_progress_top10 <- r1_info2_2p_game_progress |> 
  filter(team %in% info2_top10_teams) |> 
  mutate(team = factor(team, levels = info2_top10_teams))

info2_r1_2p_progress_top10_plot <- ggplot(data = info2_r1_2p_max_progress_top10, 
                             aes(x = game_progress, y = team, group = team, fill = team)) + 
   geom_density_ridges2(show.legend = FALSE) +
   theme_bw() +
   theme(panel.grid.major.y = element_blank(),
         panel.grid.minor.x = element_blank(),
         axis.text.x = element_text(angle = 45, hjust = 1)) +
   scale_x_continuous(breaks = progress_breaks,
                      labels = progress_labels,
                      limits = c(0, 360)) +
   scale_y_discrete(limits = rev) + 
   scale_fill_viridis_d(option = "mako", begin = 0, end = 0.85) +
   labs(x = "Progress", 
        y = "Top 10 Teams",
        title = glue("Game Progress for the Current Top 10 Cheating Strategies"))

info2_r1_2p_progress_top10_plot
# brute is good at cheating!
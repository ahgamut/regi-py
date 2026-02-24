library(tidyverse)
library(jsonlite)
library(glue)
library(boxr)

box_auth()

progress_breaks <- c(0, 20, 40, 60, 80, 
                     110, 140, 170, 200,
                    240, 280, 320, 360)

progress_labels <- c("Start", "1st Jack", "2nd Jack", "3rd Jack", "4th Jack",
                     "1st Queen", "2nd Queen", "3rd Queen", "4th Queen",
                    "1st King", "2nd King", "3rd King", "Win")

# Get parsed JSONs from Box (rowified.zip)
box_dl(file_id = 2145201325813, local_dir = tempdir(), overwrite = TRUE, pb = TRUE)
list.files(tempdir())

rowified_file_list <- unzip(glue::glue("{tempdir()}/rowified.zip"), list = TRUE)

rowified_path <- glue("{tempdir()}/rowified.zip")
sub4_path <- "sub4.csv"

sub4 <- read_csv(unz(rowified_path, sub4_path), progress = TRUE)

sub4_enemy_order <- sub4 |>
  filter(event == "STARTGAME") |>
  select(game, game.enemy_pile) |>
  filter(!is.na(game.enemy_pile)) |>
  slice_head(n = 1, by = game)

sub4_team_members <- sub4 |>
  filter(!is.na(game.active_player.strategy)) |>
  select(team, game.active_player.strategy) |>
  group_by(team, game.active_player.strategy) |>
  slice_head(n = 1) |>
  ungroup() |>
  group_by(team) |>
  summarize(teammates = str_flatten(game.active_player.strategy, collapse = " & ")) |>
  mutate(teammates = ifelse(str_detect(teammates, "&"), teammates, glue("{teammates} & {teammates}"))
  )

# Summary statistics and plots for progress
sub4_progress_df <- sub4 |>
  group_by(game, team, sim) |>
  summarize(max_progress = max(game.progress, na.rm = TRUE))

sub4_progress_summaries_team <- sub4_progress_df |>
  group_by(team) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE),
            min_progress = min(max_progress, na.rm = TRUE),
            maximum_progress = max(max_progress, na.rm = TRUE)) |>
  left_join(sub4_team_members, by = "team")

sub4_progress_summaries_team_plot_df <- sub4_progress_summaries_team |>
  select(team, mean_progress, min_progress, maximum_progress, teammates) |>
  mutate(team = reorder(factor(team), desc(mean_progress)),
         teammates = reorder(factor(teammates), desc(mean_progress))) |>
  pivot_longer(cols = c("mean_progress", "min_progress", "maximum_progress"),
               names_to = "statname",
               values_to = "value")

# Top 10 Teams
sub4_progress_summaries_team_plot_df_top10 <- sub4_progress_summaries_team_plot_df |>
  filter(team %in% head(levels(team), 10))

sub4_progress_top10_plot <- ggplot(data = sub4_progress_summaries_team_plot_df_top10,
                                   mapping = aes(x = value,
                                                 y = teammates,
                                                 group = teammates,
                                                 color = teammates)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none",
          panel.grid.major.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))+
    scale_x_continuous(breaks = progress_breaks,
                       labels = progress_labels,
                       limits = c(0, 360)) + 
    scale_y_discrete(limits = rev) + 
    scale_color_viridis_d(option = "mako", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Top 10 Teams")
ggsave("top10teams.png", plot = sub4_progress_top10_plot, width = 6, height = 4, units = "in", dpi = 1000)

# Bottom 10 Teams
sub4_progress_summaries_team_plot_df_bottom10 <- sub4_progress_summaries_team_plot_df |>
  filter(team %in% tail(levels(team), 10))

sub4_progress_bottom10_plot <- ggplot(data = sub4_progress_summaries_team_plot_df_bottom10,
                                   mapping = aes(x = value,
                                                 y = teammates,
                                                 group = teammates,
                                                 color = teammates)) +
    geom_line() +
    stat_summary(fun = "median", geom = "point", size = 6, shape = 18) + 
    stat_summary(fun = "max", geom = "point", size = 3) + 
    stat_summary(fun = "min", geom = "point", size = 3) + 
    theme_bw()+
    theme(legend.position = "none",
          panel.grid.major.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))+
    scale_x_continuous(breaks = progress_breaks,
                       labels = progress_labels,
                       limits = c(0, 360)) + 
    scale_y_discrete(limits = rev) + 
    scale_color_viridis_d(option = "mako", begin = 0, end = 0.85) +
    labs(x = "Progress", 
         y = "Bottom 10 Teams")
ggsave("bottom10teams.png", plot = sub4_progress_bottom10_plot, width = 6, height = 4, units = "in", dpi = 1000)

# Summary statistics and plots for phase count
sub4_duration_df <- sub4 |>
  group_by(game, team, sim) |>
  summarize(max_phase_count = max(game.phase_count, na.rm = TRUE))

sub4_duration_summaries_game <- sub4_duration_df |>
  group_by(game) |>
  summarize(mean_phase_count = mean(max_phase_count, na.rm = TRUE),
            var_phase_count = var(max_phase_count, na.rm = TRUE),
            sd_phase_count = sd(max_phase_count, na.rm = TRUE)) |>
  left_join(sub4_enemy_order, by = "game")
str_sub(sub4_duration_summaries_game$game.enemy_pile, start = 12, end = 12) <- "\n"
str_sub(sub4_duration_summaries_game$game.enemy_pile, start = 24, end = 24) <- "\n"

# Phase count for each enemy kill
sub4_phase_count_summaries <- sub4 |>
  filter(event == "ENEMYKILL") |>
  group_by(game, team, sim) |>
  mutate(kill_number = row_number) |>
  left_join(sub4_team_members, by = "team")

sub4_progress_summaries_game <- sub4_progress_df |>
  group_by(game) |>
  summarize(mean_progress = mean(max_progress, na.rm = TRUE),
            min_progress = min(max_progress, na.rm = TRUE),
            maximum_progress = max(max_progress, na.rm = TRUE)) |>
  left_join(sub4_enemy_order, by = "game") |>
  mutate(game = reorder(factor(game), desc(mean_progress)))

games_hardest9 <- tail(levels(sub4_progress_summaries_game$game), 9)
games_hardest9_factor <- fct_inorder(factor(games_hardest9))

kill_number_helper_df <- data.frame(kill_number = 1:12,
                                    kill_label = progress_labels[-1]) |>
  mutate(kill_label = factor(kill_label, levels = progress_labels[-1]))

sub4_enemy_order_helper_df <- sub4_enemy_order |>
  filter(game %in% games_hardest9) |>
  arrange(factor(game, levels = games_hardest9)) |>
  mutate(game = fct_inorder(game),
         enemy_order = str_sub(game.enemy_pile, start = 1, end = 23),
         enemy_order = fct_inorder(enemy_order))

sub4_boxplots_df <- sub4_phase_count_summaries |>
  filter(game %in% games_hardest9) |>
  left_join(kill_number_helper_df, by = "kill_number") |>
  left_join(sub4_enemy_order_helper_df, by = "game") |>
  mutate(game = factor(game, levels = sub4_enemy_order_helper_df$game),
         enemy_order = factor(enemy_order, levels = sub4_enemy_order_helper_df$enemy_order))

sub4_phase_count_boxplots_game_worst <- sub4_boxplots_df |>
  ggplot(aes(x = kill_label,
             y = game.phase_count,
             group = kill_label)) +
  geom_boxplot() +
  labs(x = "Progress",
       y = "Phase Count") +
  theme_bw() +
  theme(panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~enemy_order)
ggsave("boxplots.png", plot = sub4_phase_count_boxplots_game_worst, width = 6, height = 4, units = "in", dpi = 1000)

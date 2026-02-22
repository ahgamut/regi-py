# R Script to Read and Process Regicide Game JSON Files

# packages
library(jsonlite)
library(tidyverse)
library(glue)

# Functions
# Function to flatten combo column
combo_flatten <- function(combo, sep = "|"){
  combo_flat <- combo |>
    summarize(value = str_flatten(value, collapse = sep), 
              strength = sum(strength))
  return(combo_flat)
}

# Function to flatten combos column
combos_flatten <- function(combos){
  combos_flattened <- list_flatten(combos) |>
  map(possibly(combo_flatten, otherwise = data.frame(value = NA, strength = NA), quiet = TRUE)) |> 
  bind_rows() |>
  replace_na(list(value = "Yield", strength = 0)) |>
  summarize(value = str_flatten(value, collapse = ";"), 
            strength = str_flatten(strength, collapse = ";"))
  return(combos_flattened)
}

# Function to flatten columns for cards in hands
# player.cards and game.active_player.cards
flatten_player.cards <- function(player.cards){
  # Replace empty lists with the empty string
  hand <- lapply(player.cards, function(i) {
    if (is.list(i) && length(i) == 0) "" else i
  })

  flattened_hand <- list_flatten(hand)

  # Make NULL into NA
  flattened_hand[sapply(flattened_hand, is.null)] <- NA

  # Flatten strings
  flattened_hand <- flattened_hand |> 
    map(\(x) str_flatten(x, collapse = "|"))

  flattened_hand_out <- list_c(flattened_hand)
  return(flattened_hand_out)
}

# Function to parse the combos used
# Combos used
parse_combos_used <- function(combos_used_col){
  output_df <- data.frame(value = rep(NA_character_, length(combos_used_col)), 
                          strength = rep(NA_character_, length(combos_used_col)))
  for(i in 1:length(combos_used_col)){
    lengthi <- length(combos_used_col[[i]])
    if(is.null(combos_used_col[[i]])){
      output_df$value[i] <- NA
      output_df$strength[i] <- NA
    }else if(!is.null(combos_used_col[[i]]) & lengthi == 0){
      output_df$value[i] <- "IDK"
      output_df$strength[i] <- "0"
    }else{
      flattened_df <- combos_flatten(combos_used_col[[i]])
      output_df$value[i] <- flattened_df[1]
      output_df$strength[i] <- flattened_df[2]
    }
  }
  output_df <- output_df |>
    mutate(value = list_c(value),
          strength = list_c(strength)
  )
  return(output_df)
}

# Functions to flatten the card piles
# game.draw_pile, game.discard_pile, game.enemy_pile
flatten_card_pile <- function(card_pile){
  # Replace empty lists with the empty string
  pile <- lapply(card_pile, function(i) {
    if (is.list(i) && length(i) == 0) "" else i
  })

  flattened_pile <- list_flatten(pile)

  # Make NULL into NA
  flattened_pile[sapply(flattened_pile, is.null)] <- NA

  # Flatten strings
  flattened_pile <- flattened_pile |> 
    map(\(x) str_flatten(x, collapse = "|"))

  flattened_pile_out <- list_c(flattened_pile)
  return(flattened_pile_out)
}

# Function to read and parse a game JSON file
parse_game_json <- function(json_path){
# Read the raw game JSON file
raw <- jsonlite::fromJSON(json_path, flatten = TRUE)

# Flatten the combo column
flattened_combos <- raw$combo |>
  map(possibly(combo_flatten, otherwise = data.frame(value = NA, strength = NA), quiet = TRUE)) |>
  bind_rows()

# Create a new dataset called game
# We will modify the game dataset until it has the form we want
game <- bind_cols(raw, flattened_combos)
  
# Create 2 new columns (combo.value and combo.strenth)
# as a result of flattening the combo list column
game <- game |>
  select(!combo) |>
  relocate(all_of(c("value", "strength")), .after = "event") |>
  rename(combo.value = value,
         combo.strength = strength)

# Update the player.cards and game.active_player.cards columns
# to be strings instead of lists
game <- game |>
  mutate(player.cards = flatten_player.cards(player.cards = player.cards),
         game.active_player.cards = flatten_player.cards(player.cards = game.active_player.cards))

# Parse the combos used
combos_used <- parse_combos_used(game$game.used_combos)

# Create 2 new columns (used_combos.value and used_combos.strength)
# as a result of parsing the game.used_combos column
game <- bind_cols(game, combos_used)
game <- game |>
  select(!game.used_combos) |>
  relocate(all_of(c("value", "strength")), .after = "game.status") |>
  rename(used_combos.value = value,
         used_combos.strength = strength)

# Update the game.draw_pile, game.discard_pile, and game.enemy_pile columns
# to be strings instead of lists
game <- game |>
  mutate(game.draw_pile = flatten_card_pile(game.draw_pile),
         game.discard_pile = flatten_card_pile(game.discard_pile),
        game.enemy_pile = flatten_card_pile(game.enemy_pile))

# Parse the game.players column
# Get first non-null element in the game.players column, which has player info
is_null_player_info <- sapply(game$game.players, is.null)
non_null_player_indices <- which(!is_null_player_info)
player_info_index <- min(non_null_player_indices)
player_info <- game$game.players[[player_info_index]]

# Get the player strategies, which we will join with the rest of the data later
strategy <- player_info |>
  select(id, strategy)

# Keep only id, alive, and num_cards for all non-null dataframes in the list
game$game.players[non_null_player_indices] <- game$game.players[non_null_player_indices] |>
  map(\(x) x |> select(id, alive, num_cards))

# Create a new dataset for whether each player is alive and how many cards each player has
game_players_pivot <- game |>
  select(game.players) |>
  mutate(row_id = 1:nrow(game)) |>
  unnest(game.players) |>
  pivot_wider(id_cols = row_id,
              names_from = id,
              names_sep = "_player_",
              values_from = c(alive, num_cards))

# Join the dataset with player information to the game dataset
game <- game |>
  mutate(row_id = 1:nrow(game)) |>
  left_join(game_players_pivot, join_by("row_id")) |>
  select(!c(row_id, game.players)) |>
  relocate(starts_with(c("alive_player", "num_cards")), .after = "game.hand_size")

# Join the game dataset with the strategy dataset
game <- left_join(game, strategy, join_by(game.active_player_id == id)) |>
  relocate(strategy, .after = "game.active_player_id")
  
# Keep only relevant events: 
  # STARTGAME, ATTACK, DEFEND, ENEMYKILL, FULLBLOCK,
  # FAILBLOCK, DECKEMPTY, ENDGAME, POSTGAME

  game <- game |>
    filter(event %in% c("STARTGAME", "ATTACK", "DEFEND",
                        "ENEMYKILL", "FULLBLOCK", "FAILBLOCK",
                        "DECKEMPTY", "ENDGAME", "POSTGAME"))

# Return the parsed game JSON file
return(game)
}

# test_game <- parse_game_json(json_path = "game_json/regi-1769630227033.json")
# test_game2 <- parse_game_json(json_path = "game_json/regi-1769630521039.json")
# win <- parse_game_json("game_json/first-win.json")

# Function to make a diagram of a Regicide game game phase
# (i.e., one row of a parsed )
diagram_game_phase <- function(turn){

  plotdf <- turn |>
  select(event, game.active_player_id, combo.value,
    game.phase_count, player.cards,
    starts_with("num_cards_player"), used_combos.value,
    game.draw_pile_size, game.discard_pile_size,
    game.enemy_pile_size,
    game.current_enemy.value, game.current_enemy.hp
  ) |>
  rename(
    Event = event,
    `Active Player ID` = game.active_player_id,
    `Cards Discarded` = combo.value,
    `Game Phase Count` = game.phase_count,
    `Player Hand` = player.cards,
    `Cards Played` = used_combos.value,
    `Tavern Deck Size` = game.draw_pile_size,
    `Discard Pile Size` = game.discard_pile_size,
    `Enemies Left` = game.enemy_pile_size,
    `Current Enemy` = game.current_enemy.value,
    `Current Enemy HP` = game.current_enemy.hp
  ) |>
  rename_with(
    ~paste0("Number of Cards for Player ", str_sub(.x, 18)),
    .cols = starts_with("num_cards_player")
  ) |>
  mutate(across(
    c(`Game Phase Count`, `Tavern Deck Size`, `Discard Pile Size`, 
      `Enemies Left`, `Active Player ID`,
    starts_with("Number of Cards"), `Current Enemy HP`
  ), as.character),
   `Cards Played` = str_replace_all(`Cards Played`, ";", "\n"),
   `Cards Discarded` = ifelse(Event == "DEFEND", `Cards Discarded`, NA)) |>
    relocate(starts_with("Number of Cards"), .after = everything()) |>
  pivot_longer(
    cols = c(Event, `Active Player ID`, `Cards Discarded`,
    `Game Phase Count`, `Player Hand`,
    `Cards Played`,
    `Tavern Deck Size`, `Discard Pile Size`,
    `Enemies Left`, 
    `Current Enemy`, `Current Enemy HP`, 
    starts_with("Number of Cards")),
    names_to = c("labels"),
    values_to = "value"
  ) |>
  mutate(
    plot_info = 
    dplyr::case_when(labels %in% c("Cards Played", "Player Hand", "Cards Discarded") ~ glue::glue("{labels}:\n{value}"),
      .default = glue::glue("{labels}: {value}"))
    )
  
  # Add coordinates for diagram
  if (nrow(plotdf) == 13) {
    # 2 player game
    plotdf <- plotdf |>
    mutate(x = c(-2.25, 2, 2, -2.25, 2, -2.25, 0, 0, 0, -2.25, -2.25, 0, 0),
           y = c(9, 7.5, 5, 8.5, 5.5, 5.5, 7.5, 7, 6.5, 7.5, 7, 6, 5.5))
  } else if (nrow(plotdf) == 14) {
    # 3 player game
    plotdf <- plotdf |>
    mutate(x = c(-2.25, 2, 2, -2.25, 2, -2.25, 0, 0, 0, -2.25, -2.25, 0, 0, 0),
           y = c(9, 7.5, 5, 8.5, 5.5, 5.5, 7.5, 7, 6.5, 7.5, 7, 6, 5.5, 5))
  } else if (nrow(plotdf == 15)){
    # 4 player game
    plotdf <- plotdf |>
    mutate(x = c(-2.25, 2, 2, -2.25, 2, -2.25, 0, 0, 0, -2.25, -2.25, 0, 0, 0, 0),
           y = c(9, 7.5, 5, 8.5, 5.5, 5.5, 7.5, 7, 6.5, 7.5, 7, 6, 5.5, 5, 4.5))
  }
  

game_diagram <- ggplot(data = plotdf, aes(x = x, y = y, label = plot_info)) +
  geom_text() +
  theme_void() +
  scale_x_continuous(limits = c(-3, 3))

return(game_diagram)
}

# rows_to_plot <- test_game[2:11,]
# rows_to_plot_list <- split(rows_to_plot, seq(nrow(rows_to_plot)))

# rows_to_plot_list |>
#   map(\(x) diagram_game_phase(x))

# test_game_plot <- test_game[2:59,]
# test_game_plot_list <- split(test_game_plot, seq(nrow(test_game_plot)))

# test_game_diagrams <- test_game_plot_list |>
#    map(\(x) diagram_game_phase(x))

# win_list <- split(win, seq(nrow(win)))
# win_diagrams <- win_list |>
#     map(\(x) diagram_game_phase(x))

progress_breaks <- c(20, 40, 60, 80, 
                     110, 140, 170, 200,
                    240, 280, 320, 360)

progress_labels <- c("1st Jack", "2nd Jack", "3rd Jack", "4th Jack",
                     "1st Queen", "2nd Queen", "3rd Queen", "4th Queen",
                    "1st King", "2nd King", "3rd King", "Win")

# Function to make a line plot
# Input
# df: The dataframe to plot.
# group: The grouping variable for the plot.
# statistic: The variable containing the statistic to plot.
# k: The number of groups to plot.
# top: Should the top groups be plotted (TRUE) or the bottom groups (FALSE)?
# breaks: Breaks for the scale for the statistic
# labels: Labels for the scale for the statistic
# lims: Vector of scale limits for the statistic
# xlabel: The x label for the plot.
# ylabel: The y label for the plot
lineplot <- function(df, group, statistic, k = nrow(df), top = TRUE, 
                     breaks = progress_breaks, labels = progress_labels, lims = c(0, 360),
                     xlabel, ylabel){

    if (isTRUE(top)) {
      # Top k groups
      dfarr <- df |> 
        arrange(desc({{ statistic }})) |>
        head(n = k) |>
        mutate(grouparr = reorder(factor({{ group }}), {{ statistic }}))
    } else {
      # Bottom k groups
      dfarr <- df |>
        arrange(desc({{ statistic }})) |>
        tail(n = k) |>
        mutate(grouparr = reorder(factor({{ group }}), {{ statistic }}))
    }
  
  lp <- dfarr |>
    ggplot(aes(y = grouparr)) +
    geom_point(aes(x = {{ statistic }})) +
    geom_segment(aes(x = 0, xend = {{ statistic }},
                     yend = grouparr)) +
    scale_x_continuous(breaks = breaks,
                       labels = labels,
                       limits = lims) +
    labs(x = xlabel, y = ylabel) +
    theme_bw() +
    theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
  return(lp)
}


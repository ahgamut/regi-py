# R Script to Read and Process Regicide Game JSON Files
# Code from Copilot with modifications by Marie

# packages
library(jsonlite)
library(tidyverse)

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
#"game_json/regi-1769630227033.json"
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
  
# Flatten the combos column
flattened_combo_options <- game$combos |>
  map(possibly(combos_flatten, otherwise = data.frame(value = NA, strength = NA), quiet = TRUE)) |>
  bind_rows()

# Create 2 new columns (combos.value and combos.strenth)
# as a result of flattening the combos list column
game <- bind_cols(game, flattened_combo_options)
game <- game |>
  select(!combos) |>
  relocate(all_of(c("value", "strength")), .after = "userid") |>
  rename(combos.value = value,
         combos.strength = strength)

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
  
# Return the parsed game JSON file
return(game)
}

# test_game <- parse_game_json(json_path = "game_json/regi-1769630227033.json")
# test_game2 <- parse_game_json(json_path = "game_json/regi-1769630521039.json")

# Load required packages
library(tidytext)
library(dplyr)
library(stringr)
library(readr)
library(tm)
library(topicmodels)
library(ggplot2)

# Step 1: Import dataset
tweets_raw <- read_csv("data/AvengersEndGame_tweets.csv")  # place the Kaggle dataset in a data/ folder
tweets_clean <- tweets_raw %>%
  filter(!is.na(text)) %>%
  select(text)

# Step 2: Clean tweets (CamelCase split before lowercase)
tweets_clean <- tweets_clean %>%
  mutate(text = str_replace_all(text, "http\\S+|www\\S+", "")) %>%
  mutate(text = str_replace_all(text, "@\\w+|#\\w+", "")) %>%
  mutate(text = str_replace_all(text, "[^A-Za-z]", " ")) %>%
  mutate(text = str_replace_all(text, "(?<=[a-z])(?=[A-Z])", " ")) %>%
  mutate(text = str_squish(text)) %>%
  mutate(text = str_to_lower(text))

# Step 3: Define stopwords
data("stop_words")
custom_stopwords <- c(
  "rt", "avengers", "endgame", "marvel", "movie", "watch", "trailer",
  "omg", "lol", "pls", "amp", "just", "like", "one", "now", "get", "see",
  "really", "still", "im", "dont", "did", "cant", "going", "got", "day",
  "avengersendgame", "https", "t.co", "ads", "premiere", "helloboon", 
  "chrisevans", "chris", "johansson", "evans", "gif", "hmvavengersendga",
  "marvelstudios", "paytm", "robertdowneyjr", "hemsworth", "brandondavisbd",
  "fanart", "ticketbooking", "endgamereaction",
  "scarlett", "miley", "liam", "elizabeth", "olsen", "billy", "joel",
  "josh", "alan", "robert", "downey", "pratt", "cyrus", "thompson"
)

combined_stopwords <- stop_words %>%
  select(word) %>%
  distinct() %>%
  bind_rows(tibble(word = custom_stopwords)) %>%
  mutate(word = str_to_lower(word)) %>%
  distinct()

write_csv(combined_stopwords, "custom_stopwords.csv")

# Step 4: Tokenize and filter
tokens <- tweets_clean %>%
  unnest_tokens(word, text)

tokens_final <- tokens %>%
  anti_join(combined_stopwords, by = "word") %>%
  filter(str_length(word) > 2) %>%
  filter(!str_detect(word, "[0-9]")) %>%
  filter(str_detect(word, "[aeiou]")) %>%
  filter(!str_detect(word, "[^a-z]")) %>%
  mutate(document = row_number())

# Step 5: Create document-term matrix
dtm <- tokens_final %>%
  count(document, word) %>%
  cast_dtm(document, word, n)

# Step 6: Fit LDA models for multiple k values
extract_lda_model <- function(k_value, dtm, top_n = 10) {
  model <- LDA(dtm, k = k_value, control = list(seed = 123))
  
  top_terms <- tidy(model, matrix = "beta") %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n) %>%
    arrange(topic, -beta) %>%
    mutate(k = k_value)
  
  topic_distribution <- tidy(model, matrix = "gamma") %>%
    count(topic) %>%
    mutate(percent = n / sum(n) * 100, k = k_value)
  
  return(list(model = model, top_terms = top_terms, gamma = topic_distribution))
}

k_values <- c(3, 5, 7, 10)
model_results <- lapply(k_values, extract_lda_model, dtm = dtm)
names(model_results) <- paste0("k_", k_values)

# Step 7: Visualize top terms across all models
all_top_terms <- bind_rows(lapply(model_results, function(x) x$top_terms))

plot_top_terms <- all_top_terms %>%
  mutate(term = reorder_within(term, beta, interaction(k, topic))) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ interaction(k, topic), scales = "free", ncol = 2) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Terms per Topic Across k Values", x = "Term", y = "Beta")

print(plot_top_terms)
ggsave("top_terms_across_k.png", plot_top_terms, width = 10, height = 8)

# Step 8: Visualize topic distribution
gamma_summary <- bind_rows(lapply(model_results, function(x) x$gamma))
print(gamma_summary)

library(scales)
plot_topic_dist <- gamma_summary %>%
  ggplot(aes(x = factor(topic), y = percent, fill = factor(k))) +
  geom_col(position = "dodge") +
  facet_wrap(~ k, scales = "free_x") +
  labs(title = "Topic Distribution Across k Values", x = "Topic", y = "Percent") +
  scale_y_continuous(labels = percent_format())

print(plot_topic_dist)
ggsave("topic_distribution_across_k.png", plot_topic_dist, width = 10, height = 6)

# Step 9: Evaluate semantic distinctiveness via term overlap
term_overlap <- all_top_terms %>%
  count(term, k, topic) %>%
  group_by(term) %>%
  summarize(
    models_used = n_distinct(k),
    topics_used = n(),
    total_mentions = sum(n)
  ) %>%
  arrange(desc(total_mentions))

print(term_overlap)

plot_repeated_terms <- term_overlap %>%
  filter(models_used > 1) %>%
  top_n(30, total_mentions) %>%
  ggplot(aes(x = reorder(term, total_mentions), y = total_mentions)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Most Repeated Terms Across Topics and Models", x = "Term", y = "Mentions")

print(plot_repeated_terms)
ggsave("repeated_terms_across_models.png", plot_repeated_terms, width = 8, height = 6)

# Manual coherence check for all models
term_summary <- all_top_terms %>%
  group_by(k, topic) %>%
  summarize(terms = paste(term, collapse = ", "), .groups = "drop") %>%
  arrange(k, topic)

print(term_summary, n = Inf)

# Best model is k=5

top_terms_k5 <- model_results[["k_5"]]$top_terms %>%
  group_by(topic) %>%
  summarize(terms = paste(term, collapse = ", "), .groups = "drop")

topic_labels <- c(
  "Fan Anticipation & Emotional Reactions",
  "Premiere Events & Cast Tributes",
  "Ticketing, Cast Mentions & Red Carpet",
  "Spoilers & Character Moments",
  "Red Carpet & Fan Celebrations"
)

top_terms_labeled <- top_terms_k5 %>%
  mutate(label = topic_labels[topic])

print(top_terms_labeled, n = Inf)
write_csv(top_terms_labeled, "topic_labels_k5.csv")


# sentiment analysis
library(tidytext)
library(dplyr)
library(tidyr)
library(ggplot2)

tweets_clean <- tweets_clean %>%
  mutate(id = row_number())

sentiment_tokens <- tweets_clean %>%
  select(id, text) %>%
  unnest_tokens(word, text) %>%
  inner_join(get_sentiments("bing"), by = "word")

sentiment_scores <- sentiment_tokens %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative)

gamma_k5 <- tidy(model_results[["k_5"]]$model, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%
  ungroup() %>%
  mutate(id = as.integer(document))

sentiment_by_topic <- sentiment_scores %>%
  inner_join(gamma_k5, by = "id") %>%
  group_by(topic) %>%
  summarize(
    avg_sentiment = mean(sentiment_score),
    sd_sentiment = sd(sentiment_score),
    tweet_count = n()
  )

ggplot(sentiment_by_topic, aes(x = factor(topic), y = avg_sentiment)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = avg_sentiment - sd_sentiment,
                    ymax = avg_sentiment + sd_sentiment),
                width = 0.2, color = "gray30") +
  labs(
    title = "Average Sentiment Score by Topic (k = 5)",
    x = "Topic",
    y = "Average Sentiment Score"
  ) +
  theme_minimal()

write_csv(sentiment_by_topic, "sentiment_by_topic_k5.csv")

# sentiment categories
sentiment_scores <- sentiment_scores %>%
  mutate(sentiment_label = case_when(
    sentiment_score > 0 ~ "positive",
    sentiment_score == 0 ~ "neutral",
    sentiment_score < 0 ~ "negative"
  ))

sentiment_distribution <- sentiment_scores %>%
  inner_join(gamma_k5, by = "id") %>%
  count(topic, sentiment_label) %>%
  group_by(topic) %>%
  mutate(percent = n / sum(n) * 100)

ggplot(sentiment_distribution, aes(x = factor(topic), y = percent, fill = sentiment_label)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Sentiment Distribution by Topic (k = 5)",
    x = "Topic",
    y = "Proportion of Tweets",
    fill = "Sentiment"
  ) +
  theme_minimal()

# extract sample tweets from each sentiment category per topic

tweet_sentiment_topic <- tweets_clean %>%
  select(id, text) %>%
  inner_join(sentiment_scores %>% select(id, sentiment_label), by = "id") %>%
  inner_join(gamma_k5 %>% select(id, topic), by = "id")

set.seed(42)  # for reproducibility

examples <- tweet_sentiment_topic %>%
  group_by(topic, sentiment_label) %>%
  slice_sample(n = 3) %>%
  ungroup()

print(examples)

write_csv(examples, "example_tweets_by_sentiment_and_topic.csv")

# Step 1: Extract full gamma matrix for k = 5
gammaDF <- tidy(model_results[["k_5"]]$model, matrix = "gamma") %>%
  mutate(document = as.integer(document))

# Step 2: Join with tweet text
tweets_with_doc_id <- tweets_clean %>%
  mutate(document = row_number())  # Ensure document ID matches gammaDF

# Step 3: Get exactly 5 tweets per topic
top_tweets_by_topic <- gammaDF %>%
  arrange(topic, desc(gamma)) %>%
  group_by(topic) %>%
  slice_head(n = 5) %>%  # Always returns top 5, ignoring ties
  ungroup() %>%
  inner_join(tweets_with_doc_id, by = "document") %>%
  select(topic, gamma, text)

print(top_tweets_by_topic, n = Inf)  # View all 25 tweets (5 per topic)

topic_labels <- c(
  "Fan Anticipation & Emotional Reactions",
  "Premiere Events & Cast Tributes",
  "Ticketing, Cast Mentions & Red Carpet",
  "Spoilers & Character Moments",
  "Red Carpet & Fan Celebrations"
)

top_tweets_by_topic <- top_tweets_by_topic %>%
  mutate(topic_label = topic_labels[topic]) %>%
  select(topic, topic_label, gamma, text)

write_csv(top_tweets_by_topic, "top_5_tweets_per_topic.csv")  # Optional export



sentiment_preview <- sentiment_scores %>%
  inner_join(tweets_clean %>% select(id, text), by = "id") %>%
  mutate(sentiment_label = case_when(
    sentiment_score > 0 ~ "positive",
    sentiment_score == 0 ~ "neutral",
    sentiment_score < 0 ~ "negative"
  )) %>%
  arrange(id) %>%
  slice_head(n = 20)



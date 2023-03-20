library(contextual)
library(ggnormalviolin)
library(reshape2)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr,
               tidyr,
               ggplot2,
               reshape2,
               latex2exp,
               devtools,
               BiocManager)

# set the seed
set.seed(1234)

get_results <- function(res){
  return(res$data %>%
           select(t, sim, choice, reward, agent))
}


get_maxObsSims <- function(result){
  return(result%>%
           group_by(sim, agent) %>%
           summarise(max_t = max(t)))
}


show_results <- function(df_results, max_obs){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  df_history_agg <- df_results %>%
    group_by(sim)%>% # group by simulation
    mutate(cumulative_reward = cumsum(reward))%>% # calculate, per sim, cumulative reward over time
    group_by(t) %>% # group by timestep 
    summarise(avg_cumulative_reward = mean(cumulative_reward), # average cumulative reward
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>% # SE + Confidence interval
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <= max_obs)
  
  
  # define the legend of the plot
  legend <- c("Avg." = "orange", "95% CI" = "gray") # set legend
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward))+ 
    geom_line(size=1.5,aes(color="Avg."))+ # add line 
    geom_ribbon(aes(ymin=ifelse(cumulative_reward_lower_CI<0, 0,cumulative_reward_lower_CI),
                    # add confidence interval
                    ymax=cumulative_reward_upper_CI,
                    color = "95% CI"
    ), # 
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color='Metric')+ # add titles
    scale_color_manual(values=legend)+ # add legend
    theme_bw()+ # set the theme
    theme(text = element_text(size=16)) # enlarge text
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2, df_history_agg = df_history_agg))
}


show_results_multipleagents <- function(df_results, max_obs, n_sim){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  
  # data.frame aggregated for two versions: 20 and 40 arms
  df_history_agg <- df_results %>%
    group_by(agent, sim)%>% # group by number of arms, the sim
    mutate(cumulative_reward = cumsum(reward))%>% # calculate cumulative sum
    group_by(agent, t) %>% # group by number of arms, the t
    summarise(avg_cumulative_reward = mean(cumulative_reward),# calc cumulative reward, se, CI
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>%
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <=max_obs)
  
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward, color =agent))+
    geom_line(size=1.5)+
    geom_ribbon(aes(ymin=cumulative_reward_lower_CI , 
                    ymax=cumulative_reward_upper_CI,
                    fill = agent,
    ),
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color ='c', fill='c')+
    theme_bw()+
    theme(text = element_text(size=16))
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}


run_simulator <- function(agents, size_sim, n_sim){
  # Create Similator
  simulator          <- Simulator$new(agents, # set our agents
                                      horizon= size_sim, # set the sizeof each simulation
                                      do_parallel = TRUE, # run in parallel for speed
                                      simulations = n_sim, # simulate it n_sim times,
                                      
  )
  
  # run the simulator object
  history <- simulator$run()
  res <- get_results(history)
  max_obs <- min(get_maxObsSims(res)$max_t)
  res <- show_results_multipleagents(res, max_obs)
  return(res)
}


# read in dataset
df <- read.csv('./df.csv')

# Create the bandit
bandit <- OfflineReplayEvaluatorBandit$new(formula = returns ~ Symbol, data = df, randomize = FALSE, replacement = FALSE)
bandit_context <- OfflineReplayEvaluatorBandit$new(formula = returns ~ Symbol| roll_returns_week + prev_returns + obv, data = df, randomize = FALSE, replacement = FALSE)

# Define the UCB policy
UCB_0005 <- LinUCBDisjointPolicy$new(alpha = 0.005)
UCB_context_0005 <- LinUCBDisjointPolicy$new(alpha = 0.005)
UCB_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)
UCB_context_001 <- LinUCBDisjointPolicy$new(alpha = 0.01)
UCB_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_context_005 <- LinUCBDisjointPolicy$new(alpha = 0.05)
UCB_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_context_01 <- LinUCBDisjointPolicy$new(alpha = 0.1)
UCB_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_context_02 <- LinUCBDisjointPolicy$new(alpha = 0.2)
UCB_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_context_03 <- LinUCBDisjointPolicy$new(alpha = 0.3)
UCB_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_context_04 <- LinUCBDisjointPolicy$new(alpha = 0.4)
UCB_05 <- LinUCBDisjointPolicy$new(alpha = 0.5)
UCB_context_05 <- LinUCBDisjointPolicy$new(alpha = 0.5)
UCB_08 <- LinUCBDisjointPolicy$new(alpha = 0.8)
UCB_context_08 <- LinUCBDisjointPolicy$new(alpha = 0.8)
UCB_1 <- LinUCBDisjointPolicy$new(alpha = 1.0)
UCB_context_1 <- LinUCBDisjointPolicy$new(alpha = 1.0)

# Create UCB agents
agent_context_UCB_0005 <- Agent$new(policy = UCB_context_0005, bandit = bandit_context, name = "CUCB alpha=0.005")
agent_context_UCB_001 <- Agent$new(policy = UCB_context_001, bandit = bandit_context, name = "CUCB alpha=0.01")
agent_context_UCB_005 <- Agent$new(policy = UCB_context_005, bandit = bandit_context, name = "CUCB alpha=0.05")
agent_context_UCB_01 <- Agent$new(policy = UCB_context_01, bandit = bandit_context, name = "CUCB alpha=0.1")
agent_context_UCB_02 <- Agent$new(policy = UCB_context_02, bandit = bandit_context, name = "CUCB alpha=0.2")
agent_context_UCB_03 <- Agent$new(policy = UCB_context_03, bandit = bandit_context, name = "CUCB alpha=0.3")
agent_context_UCB_04 <- Agent$new(policy = UCB_context_04, bandit = bandit_context, name = "CUCB alpha=0.4")
agent_context_UCB_05 <- Agent$new(policy = UCB_context_05, bandit = bandit_context, name = "CUCB alpha=0.5")
agent_context_UCB_08 <- Agent$new(policy = UCB_context_08, bandit = bandit_context, name = "CUCB alpha=0.8")
agent_context_UCB_1 <- Agent$new(policy = UCB_context_1, bandit = bandit_context, name = "CUCB alpha=1.0")


agents <- list(agent_context_UCB_005, agent_context_UCB_03, agent_context_UCB_05, agent_context_UCB_08, agent_context_UCB_1)
#agents_context <- list()
# Simulator settings
size_sim=100000
n_sim=10

# run simulator UCB individual stocks

# Create Similator
simulator          <- Simulator$new(agents, # set our agents
                                    horizon= size_sim, # set the sizeof each simulation
                                    do_parallel = TRUE, # run in parallel for speed
                                    simulations = n_sim, # simulate it n_sim times,
                                    
)

# run the simulator object
history <- simulator$run()
res <- get_results(history)
max_obs <- min(get_maxObsSims(res)$max_t)
show_results_multipleagents(res, max_obs, n_sim)

#############################################################


# Define the TS policy
TS_0005 <- ContextualLinTSPolicy$new(v = 0.005)
TS_context_0005 <- ContextualLinTSPolicy$new(v = 0.005)
TS_001 <- ContextualLinTSPolicy$new(v = 0.01)
TS_context_001 <- ContextualLinTSPolicy$new(v = 0.01)
TS_005 <- ContextualLinTSPolicy$new(v = 0.05)
TS_context_005 <- ContextualLinTSPolicy$new(v = 0.05)
TS_01 <- ContextualLinTSPolicy$new(v = 0.1)
TS_context_01 <- ContextualLinTSPolicy$new(v = 0.1)
TS_02 <- ContextualLinTSPolicy$new(v = 0.2)
TS_context_02 <- ContextualLinTSPolicy$new(v = 0.2)
TS_03 <- ContextualLinTSPolicy$new(v = 0.3)
TS_context_03 <- ContextualLinTSPolicy$new(v = 0.3)
TS_04 <- ContextualLinTSPolicy$new(v = 0.4)
TS_context_04 <- ContextualLinTSPolicy$new(v = 0.4)
TS_05 <- ContextualLinTSPolicy$new(v = 0.5)
TS_context_05 <- ContextualLinTSPolicy$new(v = 0.5)
TS_08 <- ContextualLinTSPolicy$new(v = 0.8)
TS_context_08 <- ContextualLinTSPolicy$new(v = 0.8)
TS_1 <- ContextualLinTSPolicy$new(v = 1.0)
TS_context_1 <- ContextualLinTSPolicy$new(v = 1.0)

# Create TS agents
agent_context_TS_0005 <- Agent$new(policy = TS_0005, bandit = bandit_context, name = "CTS v=0.005")
agent_context_TS_001 <- Agent$new(policy = TS_001, bandit = bandit_context, name = "CTS v=0.01")
agent_context_TS_005 <- Agent$new(policy = TS_005, bandit = bandit_context, name = "CTS v=0.05")
agent_context_TS_01 <- Agent$new(policy = TS_01, bandit = bandit_context, name = "CTS v=0.1")
agent_context_TS_02 <- Agent$new(policy = TS_02, bandit = bandit_context, name = "CTS v=0.2")
agent_context_TS_03 <- Agent$new(policy = TS_03, bandit = bandit_context, name = "CTS v=0.3")
agent_context_TS_04 <- Agent$new(policy = TS_04, bandit = bandit_context, name = "CTS v=0.4")
agent_context_TS_05 <- Agent$new(policy = TS_05, bandit = bandit_context, name = "CTS v=0.5")
agent_context_TS_08 <- Agent$new(policy = TS_08, bandit = bandit_context, name = "CTS v=0.8")
agent_context_TS_1 <- Agent$new(policy = TS_1, bandit = bandit_context, name = "CTS v=1.0")


agents <- list(agent_context_TS_005, agent_context_TS_03, agent_context_TS_05, agent_context_TS_08, agent_context_TS_1)
#agents_context <- list()
# Simulator settings
size_sim=100000
n_sim=10

# run simulator UCB individual stocks

# Create Similator
simulator          <- Simulator$new(agents, # set our agents
                                    horizon= size_sim, # set the sizeof each simulation
                                    do_parallel = TRUE, # run in parallel for speed
                                    simulations = n_sim, # simulate it n_sim times,
                                    
)

# run the simulator object
history <- simulator$run()
res <- get_results(history)
max_obs <- min(get_maxObsSims(res)$max_t)
show_results_multipleagents(res, max_obs, n_sim)


# Define the TS policy
TS_0005 <- ContextualLinTSPolicy$new(v = 0.005)
TS_context_0005 <- ContextualLinTSPolicy$new(v = 0.005)
TS_001 <- ContextualLinTSPolicy$new(v = 0.01)
TS_context_001 <- ContextualLinTSPolicy$new(v = 0.01)
TS_005 <- ContextualLinTSPolicy$new(v = 0.05)
TS_context_005 <- ContextualLinTSPolicy$new(v = 0.05)
TS_01 <- ContextualLinTSPolicy$new(v = 0.1)
TS_context_01 <- ContextualLinTSPolicy$new(v = 0.1)
TS_02 <- ContextualLinTSPolicy$new(v = 0.2)
TS_context_02 <- ContextualLinTSPolicy$new(v = 0.2)
TS_03 <- ContextualLinTSPolicy$new(v = 0.3)
TS_context_03 <- ContextualLinTSPolicy$new(v = 0.3)
TS_04 <- ContextualLinTSPolicy$new(v = 0.4)
TS_context_04 <- ContextualLinTSPolicy$new(v = 0.4)
TS_05 <- ContextualLinTSPolicy$new(v = 0.5)
TS_context_05 <- ContextualLinTSPolicy$new(v = 0.5)
TS_1 <- ContextualLinTSPolicy$new(v = 1.0)
TS_context_1 <- ContextualLinTSPolicy$new(v = 1.0)

# create agents for different simulation sizes
agent_context_TS_05 <- Agent$new(policy = UCB_context_05, bandit = bandit_context, name = "UCB")
agent_context_UCB_05 <- Agent$new(policy = TS_context_05, bandit = bandit_context, name = "TS")

agents <- list(agent_context_TS_05, agent_context_UCB_05)
# Simulator settings
size_sim=50000
n_sim=5

# run simulator UCB individual stocks

# Create Similator
simulator          <- Simulator$new(agents, # set our agents
                                    horizon= size_sim, # set the sizeof each simulation
                                    do_parallel = TRUE, # run in parallel for speed
                                    simulations = n_sim, # simulate it n_sim times,
                                    
)

# run the simulator object
history <- simulator$run()
res <- get_results(history)
max_obs <- min(get_maxObsSims(res)$max_t)
show_results_multipleagents(res, max_obs)
max_obs

res <- run_simulator(list(agent_context_TS_05, agent_context_UCB_05), size_sim, n_sim)
res$fig1

# Create the agent
agent_UCB_04 <- Agent$new(policy = UCB_04, bandit = bandit, name = "UCB a=0.4")
agent_context_UCB_04 <- Agent$new(policy = UCB_context_04, bandit = bandit_context, name = "CUCB a=0.4")
agent_UCB_03 <- Agent$new(policy = UCB_03, bandit = bandit, name = "UCB a=0.3")
agent_context_UCB_03 <- Agent$new(policy = UCB_context_03, bandit = bandit_context, name = "CUCB a=0.3")
agent_UCB_02 <- Agent$new(policy = UCB_02, bandit = bandit, name = "UCB a=0.2")
agent_context_UCB_02 <- Agent$new(policy = UCB_context_02, bandit = bandit_context, name = "CUCB a=0.2")
agent_UCB_01 <- Agent$new(policy = UCB_01, bandit = bandit, name = "UCB a=0.1")
agent_context_UCB_01 <- Agent$new(policy = UCB_context_01, bandit = bandit_context, name = "CUCB a=0.1")
agent_UCB_005 <- Agent$new(policy = UCB_005, bandit = bandit, name = "UCB a=0.05")
agent_context_UCB_005 <- Agent$new(policy = UCB_context_005, bandit = bandit_context, name = "CUCB a=0.05")
agent_UCB_001 <- Agent$new(policy = UCB_001, bandit = bandit, name = "UCB a=0.01")
agent_context_UCB_001 <- Agent$new(policy = UCB_context_001, bandit = bandit_context, name = "CUCB 0.01")


agent_TS_04 <- Agent$new(policy = TS_04, bandit = bandit, name = "TS v=0.4")
agent_context_TS_04 <- Agent$new(policy = TS_context_04, bandit = bandit_context, name = "CTS v=0.4")
agent_TS_03 <- Agent$new(policy = TS_03, bandit = bandit, name = "TS v=0.3")
agent_context_TS_03 <- Agent$new(policy = TS_context_03, bandit = bandit_context, name = "CTS v=0.3")
agent_TS_02 <- Agent$new(policy = TS_02, bandit = bandit, name = "TS v=0.2")
agent_context_TS_02 <- Agent$new(policy = TS_context_02, bandit = bandit_context, name = "CTS v=0.2")
agent_TS_01 <- Agent$new(policy = TS_01, bandit = bandit, name = "TS v=0.1")
agent_context_TS_01 <- Agent$new(policy = TS_context_01, bandit = bandit_context, name = "CTS v=0.1")
agent_TS_005 <- Agent$new(policy = TS_005, bandit = bandit, name = "TS v=0.05")
agent_context_TS_005 <- Agent$new(policy = TS_context_005, bandit = bandit_context, name = "CTS v=0.05")
agent_TS_001 <- Agent$new(policy = TS_001, bandit = bandit, name = "TS v=0.01")
agent_context_TS_001 <- Agent$new(policy = TS_context_001, bandit = bandit_context, name = "CTS v=0.01")




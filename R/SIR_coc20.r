library(readr)
cov_20_be <- read_csv("Documents/#Master1/PI/R/cov_20_be.csv")

SIR <- function(time, state, parameters) {
  par <- as.list(c(state, parameters))
  with(par, {
    dS <- -beta * I * S / N
    dI <- beta * I * S / N - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}

# devtools::install_github("RamiKrispin/coronavirus")
library(coronavirus)
data(coronavirus)



# extract the cumulative incidence
df <- coronavirus %>%
  dplyr::filter(country == "Belgium") %>%
  dplyr::group_by(date, type) %>%
  dplyr::summarise(total = sum(cases, na.rm = TRUE)) %>%
  tidyr::pivot_wider(
    names_from = type,
    values_from = total
  ) %>%
  dplyr::arrange(date) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(active = confirmed - death - recovered) %>%
  dplyr::mutate(
    confirmed_cum = cumsum(confirmed),
    death_cum = cumsum(death),
    recovered_cum = cumsum(recovered),
    active_cum = cumsum(active)
  )

df <- cov_20_be



#write.csv(df,"/Users/julienhubar/Documents/#Master1/PI/R/cov_19_be.csv", row.names = TRUE)

# put the daily cumulative incidence numbers for Belgium from
# Feb 4 to March 30 into a vector called Infected
library(lubridate)

sir_start_date <- "2020-02-02"
sir_end_date <- "2020-02-18"

Infected <- subset(df, date >= ymd(sir_start_date) & date <= ymd(sir_end_date))$active_cum

# Create an incrementing Day vector the same length as our
# cases vector
Day <- 1:(length(Infected))

# now specify initial values for N, S, I and R
N <- 1000000
init <- c(
  S = N - Infected[1],
  I = Infected[1],
  R = 0
)

# define a function to calculate the residual sum of squares
# (RSS), passing in parameters beta and gamma that are to be
# optimised for the best fit to the incidence data
RSS <- function(parameters) {
  names(parameters) <- c("beta", "gamma")
  out <- ode(y = init, times = Day, func = SIR, parms = parameters)
  fit <- out[, 3]
  sum((Infected - fit)^2)
}

library(deSolve)

Opt <- optim(c(0.5, 0.5),
             RSS,
             method = "L-BFGS-B",
             lower = c(0, 0),
             upper = c(1, 1)
)

# check for convergence
Opt$message

Opt_par <- setNames(Opt$par, c("beta", "gamma"))
Opt_par

# time in days for predictions
t <- 1:as.integer(ymd(sir_end_date) + 1 - ymd(sir_start_date))

# get the fitted values from our SIR model
fitted_cumulative_incidence <- data.frame(ode(
  y = init, times = t,
  func = SIR, parms = Opt_par
))

# add a Date column and the observed incidence data
library(dplyr)
fitted_cumulative_incidence <- fitted_cumulative_incidence %>%
  mutate(
    Date = ymd(sir_start_date) + days(t - 1),
    Country = "Belgium",
    cumulative_incident_cases = Infected
  )

# plot the data
library(ggplot2)
fitted_cumulative_incidence %>%
  ggplot(aes(x = Date)) +
  geom_line(aes(y = I), colour = "red") +
  geom_point(aes(y = cumulative_incident_cases), colour = "blue") +
  labs(
    y = "Cumulative incidence",
    title = "COVID-20 fitted vs observed cumulative incidence, Belgium",
    subtitle = "(Red = fitted from SIR model, blue = observed)"
  ) +
  theme_minimal()

fitted_cumulative_incidence %>%
  ggplot(aes(x = Date)) +
  geom_line(aes(y = I), colour = "red") +
  geom_point(aes(y = cumulative_incident_cases), colour = "blue") +
  labs(
    y = "Cumulative incidence",
    title = "COVID-19 fitted vs observed cumulative incidence, Belgium",
    subtitle = "(Red = fitted from SIR model, blue = observed)"
  ) +
  theme_minimal() +
  scale_y_log10(labels = scales::comma)

Opt_par

R0 <- as.numeric(Opt_par[1] / Opt_par[2])
R0

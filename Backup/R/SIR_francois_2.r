
setwd("C:/Users/franc/Documents/GitHub/PI/R")
library(readr)
df <- read_csv("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv")

# devtools::install_github("RamiKrispin/coronavirus")
library(coronavirus)
data(coronavirus)

SIR <- function(time, state, parameters) {
  par <- as.list(c(state, parameters))
  with(par, {
    dS <- -beta * I * S / N
    dI <- beta * I * S / N - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}

# Make cumul
df[1, 2] <- 1
confir <- df$num_positive
cum_hospit <-df$num_cumulative_hospitalizations
Infected <- confir
for(i in(2: length(Infected))){
    Infected[i] <- Infected[i] + Infected[i-1]
}
Infected <- Infected + cum_hospit
Infected <- df$num_cumulative_hospitalizations
# Time vector
Day <- 1:(length(Infected))
# Initial state
N <- 1000000
init <- c(
  S = N - Infected[1],
  I = Infected[1],
  R = 0
)

library(deSolve)

# Least square to optimize
RSS <- function(parameters) {
  names(parameters) <- c("beta", "gamma")
  out <- ode(y = init, times = Day, func = SIR, parms = parameters)

  fit <- out[, 4]
  print(fit)
  print(" - ")
  print(Infected)
  print("\n")
  sum((Infected - out[, 4])^2)
}

predict <- function(parameters, time, initial_state){
  names(parameters) <- c("beta", "gamma")
  out <- ode(y=initial_state, time=time, func=SIR, parms=parameters)

  # plot(out)
  result <- out

}

# Test a prediction:
test_param <- c(0.5, 0.1)
test_day <- 1:100
predict(test_param, test_day, init)

# Fit:
Opt <- optim(c(0.5, 0.5),
  RSS,
  method = "L-BFGS-B",
  lower = c(0, 0),
  upper = c(1, 1)
)

# check for convergence
Opt$message
fit_param <- Opt$par

# Predict to confirme:
predictions <- predict(fit_param, Day, init)

# plot:
plot(predictions[, 4])
plot(Infected)

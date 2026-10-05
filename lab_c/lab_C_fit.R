distance_cm <- c(20, 30, 95, 175, 215, 275)

trial1_ns <- c(56.8, 57.6, 61.2, 67.6, 69.6, 72.8)
trial2_ns <- c(56.4, 57.2, 62.4, 67.2, 70.0, 73.2)
trial3_ns <- c(56.0, 56.4, 61.6, 67.2, 68.8, 73.2)

data <- data.frame(
  distance_cm,
  trial1_ns,
  trial2_ns,
  trial3_ns
)

data$mean_time_ns <- rowMeans(
  data[, c("trial1_ns", "trial2_ns", "trial3_ns")]
)

data$sd_time_ns <- apply(
  data[, c("trial1_ns", "trial2_ns", "trial3_ns")],
  1,
  sd
)

data$se_time_ns <- sqrt(mean(data$sd_time_ns^2)) / sqrt(3)

print(data)

fit <- lm(mean_time_ns ~ distance_cm, data = data)

summary(fit)

plot(
  data$distance_cm,
  data$mean_time_ns,
  pch = 19,
  cex = 0.6,
  xlab = "Mirror position x (cm)",
  ylab = "Mean time difference (ns)"
)

arrows(
  data$distance_cm,
  data$mean_time_ns - data$se_time_ns,
  data$distance_cm,
  data$mean_time_ns + data$se_time_ns,
  angle = 90,
  code = 3,
  length = 0.05
)

abline(fit, lwd = 2)

slope <- coef(fit)[2]
intercept <- coef(fit)[1]
r_squared <- summary(fit)$r.squared

slope
intercept
r_squared

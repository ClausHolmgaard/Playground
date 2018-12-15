using Plots
using ORCA

#pyplot()
plotlyjs();

x1 = [1, 2, 3]
y1 = [1, 2, 3]

plot(x1, y1, reuse=false);
title!("First plot");
gui();

x2 = [1, 2, 3]
y2 = [3, 2, 1]
plot(x2, y2, reuse=false);
title!("Second plot")
gui();

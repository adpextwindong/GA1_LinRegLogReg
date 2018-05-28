M = dlmread("data-1.txt");
for i = 1:6000
  x = M(i, :);
  imshow(reshape(x,28,28), [0 255]);
  pause(0.05)
end
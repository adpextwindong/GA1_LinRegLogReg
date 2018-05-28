M = dlmread("eigenvectors.csv");
for i = 1:10
  x = M(i, :);
  imshow(reshape(x,28,28), [0 1]);
  pause(0.5)
end

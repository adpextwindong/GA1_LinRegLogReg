M = dlmread("eigenvectors.csv");
x = M(1, :);
imshow(reshape(x,28,28), [0 1]);
pause(3)
x = M(1, :);
for i = 2:10
  x = M(i, :);
  imshow(reshape(x,28,28), [0 1]);
  pause(0.5)
end

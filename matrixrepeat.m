function [b]=matrixrepeat(a,repeat)
% a and b are linear vectors
tmp = repmat(a, repeat, 1);
b=reshape(tmp, 1, length(a)*repeat);
end
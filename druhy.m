
function[result]=druhy(filename, coords)

% priklad:
% filename = 'image.jpg';
% coords = [250 370;  % 1 [row, col]
%           220 550;  % 2
%           450 550;  % 3
%           450 350]; % 4
% vrati maticu s bodmy mimo polygonu ako 1

    I = imread(filename);
    if size(I,3) == 3
        I = rgb2gray(I);
    end

    I=im2double(I);

    rows = coords(:,1);
    cols = coords(:,2);

    rmin = min(rows); 
    rmax = max(rows);
    cmin = min(cols); 
    cmax = max(cols);

    % stvorec
    cropped = I(rmin:rmax, cmin:cmax);

    % shift
    shiftedRows = rows - rmin + 1;
    shiftedCols = cols - cmin + 1;

    [cc, rr] = meshgrid(1:size(cropped,2), 1:size(cropped,1));

    inside = inpolygon(cc, rr, shiftedCols, shiftedRows);

    % initializacia
    result = ones(size(cropped));

    % final
    result(inside) = cropped(inside);

    % display
    % figure;
    % imshow(I); title('Full Image');
    % hold on; plot([cols; cols(1)], [rows; rows(1)], 'r-', 'LineWidth', 1); % polygon
    % 
    % figure;
    % imshow(result); title('Extracted Polygon');

end
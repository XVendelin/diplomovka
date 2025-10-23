% filename = 'image.jpg';    
% startCoord = [250, 350];    % (y, x)
% stopCoord  = [450, 550];    % (y, x)

function[subMatrix] = prvy(filename, startCoord, stopCoord)
    I = imread(filename);


    if size(I,3) == 3
        I = rgb2gray(I);
    end


    r1 = min(startCoord(1), stopCoord(1));
    r2 = max(startCoord(1), stopCoord(1));
    c1 = min(startCoord(2), stopCoord(2));
    c2 = max(startCoord(2), stopCoord(2));

    I=im2double(I);

    subMatrix = I(r1:r2, c1:c2);


    figure;
    subplot(1,2,1); imshow(I); title('Full Image');
    hold on; rectangle('Position',[c1 r1 (c2-c1) (r2-r1)],'EdgeColor','r');
    subplot(1,2,2); imshow(subMatrix); title('Extracted Region');
end
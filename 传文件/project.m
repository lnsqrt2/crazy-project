map = zeros(4,9);
for i=1:36
    load("fai"+num2str(i)+".mat");
    x = round(H1*1e7);
    y = round((1.1-fai)*10);
    map(x,y) = eff_1;
end
V(map);

function v=V(M)
    M = abs(M);
    heatmap(M')
    grid off
    colormap(jet)

    h=gca;
    set(gca,'Xlabel','H1(nm)');
    set(gca,'Ylabel','Fill factor');
    h.Title = 'Efficiency';
   
    YourYticklabel=cell(size(h.YDisplayLabels));
    for i=1:9
        [YourYticklabel{i}]=deal(num2str(1.1-0.1*i));
    end
    h.YDisplayLabels=YourYticklabel;
    
    YourXticklabel=cell(size(h.XDisplayLabels));
    for i=1:4
        [YourXticklabel{i}]=deal(num2str(100*i));
    end
    h.XDisplayLabels=YourXticklabel;
end
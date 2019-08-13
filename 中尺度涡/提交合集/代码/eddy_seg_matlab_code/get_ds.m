function [ds,as] = get_ds(eddys,ssh,sst,u,v,n,siz)
    if(nargin < 7)%默认尺寸128*128
        siz = 128;
    end
    area = get_area(isnan(ssh),siz,0.2);%将陆地占比大于0.2的区域标出以便剔除
    %将无效区域数据标成0
    ssh(isnan(ssh)) = 0;%海表高度
    sst(isnan(sst)) = 0;
    u(isnan(u)) = 0;
    v(isnan(v)) = 0;
    %涡标记在图像上
    em = zeros(size(area));
    for i = 1:length(eddys)
        plist = eddys(i).Stats.PixelIdxList;%此处表示像素编号
        for j = plist
            em(j) = i;
        end
    end
    ds  = zeros(siz,siz,4,n);as  = zeros(siz,siz,n);
    %随机进行n次取集
    for k = 1:n
        de = true(1,length(eddys));
        [n,m] = size(area);
        x = randi(m);
        y = randi(n);
        while(~area(y,x))%随机生成一个满足条件的区域
            x = randi(m);y = randi(n);
        end
        %标记边界涡
        disp(x);
        disp('-----xy-----')
        disp(y);
        for i = x-1:x+siz
            disp(y);
            disp('---yi---')
            disp(i);
            if(0<(y-1) && em(y-1,i))
                de(em(y-1,i)) = false;
            end
            if(y+siz<n && em(y+siz,i))
                de(em(y+siz,i)) = false;
            end
        end
        for i = y-1:y+siz
            if(0<x-1 && em(i,x-1))
                de(em(i,x-1)) = false;
            end
            if(x+siz<m && em(i,x+siz))
                de(em(i,x+siz)) = false;
            end
        end
        %将数据复制到数据集中
        ds(:,:,1,k) = ssh(y:y+siz-1,x:x+siz-1);
        ds(:,:,2,k) = sst(y:y+siz-1,x:x+siz-1);
        ds(:,:,3,k) = u(y:y+siz-1,x:x+siz-1);
        ds(:,:,4,k) = v(y:y+siz-1,x:x+siz-1);
        %生成答案集并剔除边界涡
        vis = false(1,length(eddys));
        for i = 1:siz
            for j = 1:siz
                if(em(y+i-1,x+j-1)&&de(em(y+i-1,x+j-1)))
                    as(i,j,k) = eddys(em(y+i-1,x+j-1)).Cyc;
                    if(~vis(em(y+i-1,x+j-1)))
                        p = em(y+i-1,x+j-1);
                        vis(p) = true;
                    end
                end
            end
        end
    end
end
function [map] = get_area(map,len,rate)%标出陆地比例小于rate的区域
map = int32(map);
[n,m] = size(map);
r = len*len*rate;
for i = n-1:-1:1
    for j = m-1:-1:1
        map(i,j) =  map(i,j)+map(i+1,j)+map(i,j+1)-map(i+1,j+1);
    end
end

for i = 1:n-128
    for j = 1:m-128
        map(i,j) = map(i,j)-map(i+len,j)-map(i,j+len)+map(i+len,j+len);
    end
end

map = map<=r;
for i = 1:n
    for j = 1:m
        if(n-i<128||m-j<128)
            map(i,j) = false;
        end
    end
end
end
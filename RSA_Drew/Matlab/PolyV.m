function polyvals = PolyV(degree, num_points)

polyvals=zeros(num_points,1);

if (degree < 2) or (degree > 5)
    stop
end

if degree + 1 > num_points
    stop
end


%this section should take the num of points and make it the closest even
%number less than num_points, but it appears to do the opposite in original
%program

if mod(num_points,2)~=0
    num_points=num_points-1;
end

n3 = num_points/2;
a0 = num_points;
a2 = 0;
a4 = 0;
a6 = 0;
a8 = 0;
for i=1:n3
    aI = i;
    a2 = a2 + 2 *aI ^2;
    a4 = a4 + 2 *aI ^4;
    a6 = a6 + 2 *aI ^6;
    a8 = a8 + 2 *aI ^8;
end

if degree > 3
    den = a0 * a4 * a8 + 2 * a2 * a4 * a6 -a4 ^3 -a0 *a6 *a6 -a2*a2*a8;
    den = 1./den;
    c1 = a4*a8 - a6*a6;
    c2 = a4 *a6 -a2*a8;
    c3 = a2*a6 - a4*a4;
    aJ = -n3 -1;
    for i=1:num_points
        aJ = aJ+1;
        polyvals(i)=den*(c1+c2 *aJ *aJ +c3 *aJ .^4);
    end
end

if degree <= 3
    aJ = -n3-1;
    den = a0 *a4 -a2 *a2;
    den = 1./den;
    for i = 1:num_points
        aJ = aJ +1;
        aJ2 = aJ * aJ;
        polyvals(i)=den*(a4-aJ2*a2);
    end
end

for i=1:n3
    polyvals(i)=polyvals(i);
end

for i=n3+1:num_points+1
    polyvals(i)=polyvals(num_points +2 - i);
end

polyvals=polyvals./sum(polyvals);
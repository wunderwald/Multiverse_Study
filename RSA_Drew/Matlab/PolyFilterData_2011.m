function [ filtered_data, trend ] = PolyFilterData_2011( data, poly_size )
polynomial = PolyV(3,poly_size);

trend = conv(data,polynomial, 'valid');
filtered_data = data(floor(poly_size/2)+1:length(data)-(floor(poly_size/2)))  - trend;
% data, less convolution result = residual

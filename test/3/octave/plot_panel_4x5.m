function r_val = plot_panel_4x5 ...
     ( ...
       cfreq, ...
       dflopspercycle, ...
       nth, ...
       thr_str, ...
       dirpath, ...
       arch_str, ...
       vend_str, ...
       with_eigen ...
     )

impl = 'octave';
%impl = 'matlab';

%sp = 'default';
subp = 'tight';

if strcmp( subp, 'default' )
	position  = [100 100 2000 1500];
	papersize = [14.2 19.0];
	leg_pos_st = [3.40 8.70 1.9 1.0 ]; % (0,2br)
	leg_pos_mt = [13.08 13.09 1.9 1.0 ]; % (0,3tr)
	sp_margins = [ 0.070 0.049 ];
else
	position  = [100 100 1864 1540];
	papersize = [15.6 19.4];
	leg_pos_st = [1.15 8.70 2.1 1.2 ]; % (dgemm)
	leg_pos_mt = [12.20 13.60 2.1 1.2 ]; % (strmm)
	sp_margins = [ 0.068 0.051 ];
end

%fig = figure('Position', [100, 100, 2000, 1500]);
fig = figure('Position', position);
orient( fig, 'portrait' );
set(gcf,'PaperUnits', 'inches');
if strcmp( impl, 'octave' )
	%set(gcf,'PaperSize', [14.2 19.0]);
	set(gcf,'PaperSize', papersize);
	%set(gcf,'PaperPositionMode','auto');
	set(gcf,'PaperPositionMode','auto');
else % impl == 'matlab'
	set(gcf,'PaperSize', [13 20.0]);
	set(gcf,'PaperPosition', [0 0 13 20.0]);
	set(gcf,'PaperPositionMode','manual');
end
set(gcf,'PaperOrientation','landscape');

% Create filename "templates" for the files that contain the performance
% results.
filetemp_blis = '%s/output_%s_%s_asm_blis.m';
filetemp_open = '%s/output_%s_%s_openblas.m';
filetemp_eige = '%s/output_%s_%s_eigen.m';
filetemp_vend = '%s/output_%s_%s_vendor.m';

% Create a variable name "template" for the variables contained in the
% files outlined above.
vartemp = 'data_%s_%s_%s( :, : )';

% Define the datatypes and operations we will be plotting.
dts = [ 's' 'd' 'c' 'z' ];
ops( 1, : ) = 'gemm';
ops( 2, : ) = 'hemm';
ops( 3, : ) = 'herk';
ops( 4, : ) = 'trmm';
ops( 5, : ) = 'trsm';

% Generate datatype-specific operation names from the set of operations
% and datatypes.
opnames = gen_opnames( ops, dts );
n_opnames = size( opnames, 1 );

% Iterate over the list of datatype-specific operation names.
for opi = 1:n_opnames
%for opi = 1:1

	% Grab the current datatype combination.
	opname = opnames( opi, : );

	str = sprintf( 'Plotting %d: %s', opi, opname ); disp(str);

	% Construct filenames for the data files from templates.
	file_blis = sprintf( filetemp_blis, dirpath, thr_str, opname );
	file_open = sprintf( filetemp_open, dirpath, thr_str, opname );
	file_vend = sprintf( filetemp_vend, dirpath, thr_str, opname );

	% Load the data files.
	%str = sprintf( '  Loading %s', file_blis ); disp(str);
	run( file_blis )
	%str = sprintf( '  Loading %s', file_open ); disp(str);
	run( file_open )
	%str = sprintf( '  Loading %s', file_vend ); disp(str);
	run( file_vend )

	% Construct variable names for the variables in the data files.
	var_blis = sprintf( vartemp, thr_str, opname, 'asm_blis' );
	var_open = sprintf( vartemp, thr_str, opname, 'openblas' );
	var_vend = sprintf( vartemp, thr_str, opname, 'vendor' );

	% Use eval() to instantiate the variable names constructed above,
	% copying each to a simplified name.
	data_blis = eval( var_blis ); % e.g. data_st_sgemm_asm_blis( :, : );
	data_open = eval( var_open ); % e.g. data_st_sgemm_openblas( :, : );
	data_vend = eval( var_vend ); % e.g. data_st_sgemm_vendor( :, : );

	% Only read Eigen data in select cases.
	if with_eigen == 1
		opname_u = opname; opname_u(1) = '_';
		if nth == 1 || strcmp( opname_u, '_gemm' )
			file_eige = sprintf( filetemp_eige, dirpath, thr_str, opname );
			run( file_eige )
			var_eige = sprintf( vartemp, thr_str, opname, 'eigen' );
			data_eige = eval( var_eige ); % e.g. data_st_sgemm_eigen( :, : );
		else
			data_eige(1,1) = -1;
		end
	else
		data_eige(1,1) = -1;
	end

	% Plot one result in an m x n grid of plots, via the subplot()
	% function.
	plot_l3_perf( opname, ...
	              data_blis, ...
	              data_open, ...
	              data_eige, ...
	              data_vend, vend_str, ...
	              nth, ...
	              4, 5, ...
	              with_eigen, ...
	              cfreq, ...
	              dflopspercycle, ...
	              opi, ...
	              leg_pos_st, leg_pos_mt, ...
	              sp_margins );

end


% Construct the name of the file to which we will output the graph.
outfile = sprintf( 'l3_perf_%s_nt%d.pdf', arch_str, nth );

% Output the graph to pdf format.
if strcmp( impl, 'octave' )
	print( gcf, outfile );
else
	print( gcf, outfile, '-bestfit', '-dpdf' );
end


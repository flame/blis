function r_val = plot_panel_4x5( cfreq, ...
                                 dflopspercycle, ...
                                 nth, ...
                                 dirpath )

%cfreq = 1.8;
%dflopspercycle = 32;

% Create filename "templates" for the files that contain the performance
% results.
filetemp_blis = '%s/output_%s_%s_asm_blis.m';
filetemp_open = '%s/output_%s_%s_openblas.m';
filetemp_mkl  = '%s/output_%s_%s_mkl.m';

% Create a variable name "template" for the variables contained in the
% files outlined above.
vartemp = 'data_%s_%s_%s( :, : )';

if nth == 1
	thr_str = 'st';
else
	thr_str = 'mt';
end

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

%fig = figure;
%fig = figure('Position', [100, 100, 1600, 1200]);
fig = figure('Position', [100, 100, 2000, 1500]);
orient( fig, 'portrait' );
%set(gcf,'Position',[0 0 2000 900]);
set(gcf,'PaperUnits', 'inches');
%set(gcf,'PaperSize', [16 12.4]);
%set(gcf,'PaperPosition', [0 0 16 12.4]);
set(gcf,'PaperSize', [11 15.0]);
set(gcf,'PaperPosition', [0 0 11 15.0]);
%set(gcf,'PaperPositionMode','auto');
set(gcf,'PaperPositionMode','manual');
set(gcf,'PaperOrientation','landscape');

% Iterate over the list of datatype-specific operation names.
for opi = 1:n_opnames
%for opi = 1:1

	% Grab the current datatype combination.
	opname = opnames( opi, : );

	str = sprintf( 'Plotting %d: %s', opi, opname ); disp(str);

	% Construct filenames for the data files from templates.
	file_blis = sprintf( filetemp_blis, dirpath, thr_str, opname );
	file_open = sprintf( filetemp_open, dirpath, thr_str, opname );
	file_mkl  = sprintf( filetemp_mkl,  dirpath, thr_str, opname );

	% Load the data files.
	%str = sprintf( '  Loading %s', file_blis ); disp(str);
	run( file_blis )
	%str = sprintf( '  Loading %s', file_open ); disp(str);
	run( file_open )
	%str = sprintf( '  Loading %s', file_mkl  ); disp(str);
	run( file_mkl  )

	% Construct variable names for the variables in the data files.
	var_blis = sprintf( vartemp, thr_str, opname, 'asm_blis' );
	var_open = sprintf( vartemp, thr_str, opname, 'openblas' );
	var_mkl  = sprintf( vartemp, thr_str, opname, 'mkl' );

	% Use eval() to instantiate the variable names constructed above,
	% copying each to a simplified name.
	data_blis = eval( var_blis ); % e.g. data_st_sgemm_asm_blis( :, : );
	data_open = eval( var_open ); % e.g. data_st_sgemm_openblas( :, : );
	data_mkl  = eval( var_mkl  ); % e.g. data_st_sgemm_mkl( :, : );

	% Plot one result in an m x n grid of plots, via the subplot()
	% function.
	plot_l3_perf( opname, ...
	              data_blis, ...
	              data_open, ...
	              data_mkl, ...
	              nth, ...
	              4, 5, ...
	              cfreq, ...
	              dflopspercycle, ...
	              opi );

end

% Construct the name of the file to which we will output the graph.
outfile = sprintf( 'l3_perf_panel_nt%d', nth );

% Output the graph to pdf format.
print(gcf, outfile,'-bestfit','-dpdf');
%print(gcf, 'gemm_md','-fillpage','-dpdf');


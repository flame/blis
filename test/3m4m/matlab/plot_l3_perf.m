function r_val = plot_l3_perf( opname, ...
                               data_blis, ...
                               data_open, ...
                               data_mkl, ...
                               nth, ...
                               rows, cols, ...
                               cfreq, ...
                               dfps, ...
                               theid )

if 1
ax1 = subplot( rows, cols, theid );
hold( ax1, 'on' );
end

% Set line properties.
color_blis = 'k'; lines_blis = '-';  markr_blis = '';
color_open = 'r'; lines_open = '--'; markr_open = 'o';
color_mkl  = 'b'; lines_mkl  = '--'; markr_mkl  = '.';

% Compute the peak performance in terms of the number of double flops
% executable per cycle and the clock rate.
if opname(1) == 's' || opname(1) == 'c'
	flopspercycle = dfps * 2;
else
	flopspercycle = dfps;
end
max_perf_core = (flopspercycle * cfreq) * 1;

% Adjust title for real domain hemm and herk.
title_opname = opname;
if opname(1) == 's' || opname(1) == 'd' 
	if strcmp( extractAfter( opname, 1 ), 'hemm' ) || ...
	   strcmp( extractAfter( opname, 1 ), 'herk' )
		title_opname(2:3) = 'sy';
	end
end

% Print the title to a string.
titlename = '%s';
titlename = sprintf( titlename, title_opname );

% Set the legend strings.
blis_legend = sprintf( 'BLIS' );
open_legend = sprintf( 'OpenBLAS' );
mkl_legend  = sprintf( 'MKL' );

% Determine the final dimension.
%n_points = size( data_blis, 1 );
%x_end = data_blis( n_points, 1 );

% Set axes range values.
y_scale = 1.00;
x_begin = 0;
x_end   = data_blis( size( data_blis, 1 ), 1 );
y_begin = 0;
y_end   = max_perf_core * y_scale;

% Set axes names.
xaxisname = '     m = n = k';
if nth == 1
	yaxisname = 'GFLOPS';
else
	yaxisname = 'GFLOPS/core';
end


%flopscol = 4;
flopscol = size( data_blis, 2 );
msize = 5;
if 1
fontsize = 13;
else
fontsize = 16;
end
linesize = 0.5;
legend_loc = 'southeast';

% --------------------------------------------------------------------

x_axis( :, 1 ) = data_blis( :, 1 );

data_peak( 1, 1:2 ) = [     0 max_perf_core ];
data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

blis_ln  = line( x_axis( :, 1 ), data_blis( :, flopscol ) / nth, ...
                 'Color',color_blis, 'LineStyle',lines_blis, ...
                 'LineWidth',linesize );
open_ln  = line( x_axis( :, 1 ), data_open( :, flopscol ) / nth, ...
                 'Color',color_open, 'LineStyle',lines_open, ...
                 'LineWidth',linesize );
mkl_ln   = line( x_axis( :, 1 ), data_mkl( :, flopscol ) / nth, ...
                 'Color',color_mkl, 'LineStyle',lines_mkl, ...
                 'LineWidth',linesize );


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if x_end == 10000 || x_end == 8000
	x_tick2 = x_end - 2000;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
end

if rows == 4 && cols == 5 && ...
   theid == 5
	if nth == 1
		leg = legend( ...
		[ ...
		  blis_ln ...
		  open_ln ...
		  mkl_ln ...
		], ...
		blis_legend, ...
		open_legend, ...
		mkl_legend, ...
		'Location', legend_loc );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-3 );
		set( leg,'Units','inches' );
		%set( leg,'Position',[3.15 10.2 0.7 0.3 ] ); % 1600 1200
		set( leg,'Position',[4.20 12.7 0.7 0.3 ] ); % 2000 1500
	else
		leg = legend( ...
		[ ...
		  blis_ln ...
		  open_ln ...
		  mkl_ln ...
		], ...
		blis_legend, ...
		open_legend, ...
		mkl_legend, ...
		'Location', legend_loc );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-3 );
		set( leg,'Units','inches' );
		%set( leg,'Position',[3.15 10.2 0.7 0.3 ] ); % 1600 1200
		set( leg,'Position',[17.60 14.30 0.7 0.3 ] ); % 2000 1500
	end

end

set( ax1,'FontSize',fontsize );
set( ax1,'TitleFontSizeMultiplier',1.0 ); % default is 1.1.
box( ax1, 'on' );

titl = title( titlename );
set( titl, 'FontWeight', 'normal' ); % default font style is now 'bold'.

tpos = get( titl, 'Position' ); % default is to align across whole figure, not box.
%tpos(1) = tpos(1) + 100;
tpos(1) = tpos(1) + 40;
set( titl, 'Position', tpos ); % here we nudge it back to centered with box.

if theid > (rows-1)*cols
xlab = xlabel( ax1,xaxisname );
%tpos = get( xlab, 'Position' )
%tpos(2) = tpos(2) + 10;
%set( xlab, 'Position', tpos );
end

if mod(theid-1,cols) == 0
ylab = ylabel( ax1,yaxisname );
end

%export_fig( filename, colorflag, '-pdf', '-m2', '-painters', '-transparent' );
%saveas( fig, filename_png );

%hold( ax1, 'off' );

r_val = 0;

end

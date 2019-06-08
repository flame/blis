function r_val = plot_l3_perf( opname, ...
                               data_blis, ...
                               data_open, ...
                               data_eige, ...
                               data_vend, vend_str, ...
                               nth, ...
                               rows, cols, ...
                               with_eigen, ...
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
color_eige = 'm'; lines_eige = '-.'; markr_eige = 'x';
color_vend = 'b'; lines_vend = '-.'; markr_vend = '.';

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
%	if strcmp( extractAfter( opname, 1 ), 'hemm' ) || ...
%	   strcmp( extractAfter( opname, 1 ), 'herk' )
%		title_opname(2:3) = 'sy';
%	end
	opname_u = opname; opname_u(1) = '_';
	if strcmp( opname_u, '_hemm' ) || ...
	   strcmp( opname_u, '_herk' )
		title_opname(2:3) = 'sy';
	end
end

% Print the title to a string.
titlename = '%s';
titlename = sprintf( titlename, title_opname );

% Set the legend strings.
blis_legend = sprintf( 'BLIS' );
open_legend = sprintf( 'OpenBLAS' );
eige_legend = sprintf( 'Eigen' );
%vend_legend  = sprintf( 'MKL' );
%vend_legend  = sprintf( 'ARMPL' );
vend_legend = vend_str;

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
if data_eige(1,1) ~= -1
eige_ln  = line( x_axis( :, 1 ), data_eige( :, flopscol ) / nth, ...
                 'Color',color_eige, 'LineStyle',lines_eige, ...
                 'LineWidth',linesize );
else
eige_ln  = line( nan, nan, ...
                 'Color',color_eige, 'LineStyle',lines_eige, ...
                 'LineWidth',linesize );
end
vend_ln  = line( x_axis( :, 1 ), data_vend( :, flopscol ) / nth, ...
                 'Color',color_vend, 'LineStyle',lines_vend, ...
                 'LineWidth',linesize );


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if     6000 <= x_end && x_end < 10000
	x_tick2 = x_end - 2000;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
elseif 4000 <= x_end && x_end < 6000
	x_tick2 = x_end - 1000;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
elseif 2000 <= x_end && x_end < 3000
	x_tick2 = x_end - 400;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
end

if rows == 4 && cols == 5

	if nth == 1 && theid == 3
		if with_eigen == 1
			leg = legend( [ blis_ln open_ln eige_ln vend_ln ], ...
			              blis_legend, open_legend, eige_legend, vend_legend, ...
			              'Location', legend_loc );
		else
			leg = legend( [ blis_ln open_ln         vend_ln ], ...
			              blis_legend, open_legend,              vend_legend, ...
			              'Location', legend_loc );
		end
		set( leg,'Box','off','Color','none','Units','inches','FontSize',fontsize-3 );
		set( leg,'Position',[11.20 12.81 0.7 0.3 ] ); % (0,2br)
		%set( leg,'Position',[ 4.20 12.81 0.7 0.3 ] ); % (0,0br)
	elseif nth > 1 && theid == 4
		if with_eigen == 1
			leg = legend( [ blis_ln open_ln eige_ln vend_ln ], ...
			              blis_legend, open_legend, eige_legend, vend_legend, ...
			              'Location', legend_loc );
		else
			leg = legend( [ blis_ln open_ln         vend_ln ], ...
			              blis_legend, open_legend,              vend_legend, ...
			              'Location', legend_loc );
		end
		set( leg,'Box','off','Color','none','Units','inches','FontSize',fontsize-3 );
		%set( leg,'Position',[7.70 12.81 0.7 0.3 ] ); % (0,1br)
		%set( leg,'Position',[11.20 12.81 0.7 0.3 ] ); % (0,2br)
		set( leg,'Position',[10.47 14.17 0.7 0.3 ] ); % (0,2tl)
	end
end
		%set( leg,'Position',[ 4.20 12.75 0.7 0.3 ] ); % (0,0br)
		%set( leg,'Position',[ 7.70 12.75 0.7 0.3 ] ); % (0,1br)
		%set( leg,'Position',[10.47 14.28 0.7 0.3 ] ); % (0,2tl)
		%set( leg,'Position',[11.20 12.75 0.7 0.3 ] ); % (0,2br)
		%set( leg,'Position',[13.95 14.28 0.7 0.3 ] ); % (0,3tl)
		%set( leg,'Position',[14.70 12.75 0.7 0.3 ] ); % (0,3br)
		%set( leg,'Position',[17.45 14.28 0.7 0.3 ] ); % (0,4tl)
		%set( leg,'Position',[18.22 12.75 0.7 0.3 ] ); % (0,4br)

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

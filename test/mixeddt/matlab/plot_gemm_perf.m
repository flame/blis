function r_val = plot_gemm_perf( dt_str, ...
                                 data_blis, ...
                                 data_blis_ref, ...
                                 data_open, ...
                                 data_open_ref, ...
                                 is_mt, ...
                                 theid )

if 1
ax1 = subplot( 8, 16, theid );
hold( ax1, 'on' );
end

color_blis_ref = 'b'; lines_blis_ref = ':'; markr_blis_ref = '';
color_open_ref = 'k'; lines_open_ref = ':'; markr_open_ref = 'o';
color_mkl_ref  = 'r'; lines_mkl_ref  = ':'; markr_mkl_ref  = '.';

color_blis = 'b'; lines_blis = '-'; markr_blis = '';
color_open = 'k'; lines_open = '-'; markr_open = 'o';
color_mkl  = 'r'; lines_mkl  = '-'; markr_mkl  = '.';

if dt_str(4) == 's'
	flopspercycle = 32;
else
	flopspercycle = 16;
end

if is_mt == 1
	titlename     = '%sgemm';
	yaxisname     = 'GFLOPS/core';
	filename_pdf  = 'fig_%sgemm_m1p_k1p_n1p_has_mt_perf.pdf';
	filename_png  = 'fig_%sgemm_m1p_k1p_n1p_has_mt_perf.png';
	nth           = 4;
	x_end         = 4000;
	max_perf_core = (flopspercycle * 3.6) * 1;
else
	titlename     = '%sgemm';
	yaxisname     = 'GFLOPS';
	filename_pdf  = 'fig_%sgemm_m1p_k1p_n1p_has_st_perf.pdf';
	filename_png  = 'fig_%sgemm_m1p_k1p_n1p_has_st_perf.png';
	nth           = 1;
	x_end         = 2000;
	max_perf_core = (flopspercycle * 3.6) * 1;
end

titlename    = sprintf( titlename, dt_str );
filename_pdf = sprintf( filename_pdf, dt_str );
filename_png = sprintf( filename_png, dt_str );

%dt0_str = [ dt_str(4), dt_str(4), dt_str(4), dt_str(4) ];
dt0_str = dt_str(4);

blis_sref_legend = sprintf( 'BLIS [sc]gemm' );
blis_dref_legend = sprintf( 'BLIS [dz]gemm' );
blis_legend      = sprintf( 'BLIS mixed' );
open_sref_legend = sprintf( 'OBLA [sc]gemm' );
open_dref_legend = sprintf( 'OBLA [dz]gemm' );
open_legend      = sprintf( 'OBLA mixed' );

y_scale   = 1.00;

%xaxisname = 'problem size (m = n = k)';
xaxisname = '     m = n = k';

colorflag = '-rgb';

x_begin = 0;

y_begin = 0;
y_end   = max_perf_core * y_scale;

flopscol = 4;
msize = 5;
if 1
fontsize = 12;
else
fontsize = 16;
end
linesize = 0.7;
legend_loc = 'SouthEast';

% --------------------------------------------------------------------

%fig = figure;
%hold on; ax1 = gca;

x_axis( :, 1 ) = data_blis( :, 1 );

data_peak( 1, 1:2 ) = [     0 max_perf_core ];
data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

blis_ref = line( x_axis( :, 1 ), data_blis_ref( :, flopscol ) / nth, ...
                 'Color',color_blis_ref, 'LineStyle',lines_blis_ref, ...
                 'LineWidth',linesize );
blis_md  = line( x_axis( :, 1 ), data_blis( :, flopscol ) / nth, ...
                 'Color',color_blis, 'LineStyle',lines_blis, ...
                 'LineWidth',linesize );
open_ref = line( x_axis( :, 1 ), data_open_ref( :, flopscol ) / nth, ...
                 'Color',color_open_ref, 'LineStyle',lines_open_ref, ...
                 'LineWidth',linesize );
open_md  = line( x_axis( :, 1 ), data_open( :, flopscol ) / nth, ...
                'Color',color_open, 'LineStyle',lines_open, ...
                 'LineWidth',linesize );
%hold on; ax1 = gca;
                %'Parent',ax1, ...


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if theid == 1
leg = legend( ...
[ ...
  blis_ref ...
  blis_md ...
  open_ref ...
  open_md ...
], ...
blis_sref_legend, ...
blis_legend, ...
open_sref_legend, ...
open_legend, ...
'Location', 'best' );
%'Location', legend_loc );
set( leg,'Box','off' );
set( leg,'Color','none' );
set( leg,'FontSize',fontsize-2 );
set( leg,'Units','inches' );
elseif theid == 9
leg = legend( ...
[ ...
  blis_ref ...
  blis_md ...
  open_ref ...
  open_md ...
], ...
blis_dref_legend, ...
blis_legend, ...
open_dref_legend, ...
open_legend, ...
'Location', 'best' );
%'Location', legend_loc );
set( leg,'Box','off' );
set( leg,'Color','none' );
set( leg,'FontSize',fontsize-2 );
set( leg,'Units','inches' );

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

if theid > 112
xlab = xlabel( ax1,xaxisname );
%tpos = get( xlab, 'Position' )
%tpos(2) = tpos(2) + 10;
%set( xlab, 'Position', tpos );
end

if mod(theid-1,16) == 0
ylab = ylabel( ax1,yaxisname );
end


%export_fig( filename, colorflag, '-pdf', '-m2', '-painters', '-transparent' );
%saveas( fig, filename_png );

%hold( ax1, 'off' );

r_val = 0;

end

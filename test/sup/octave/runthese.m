% kabylake
plot_panel_trxsh(3.80,16,1,'st','d','rrr',[ 6 8 4 ],'lds','uaub','../results/kabylake/20200302/mnkt100000_st','kbl','MKL','octave'); close; clear all;

% haswell
plot_panel_trxsh(3.5,16,1,'st','d','rrr',[ 6 8 4 ],'lds','uaub','../results/haswell/20200302/mnkt100000_st','has','MKL','octave'); close; clear all;

% epyc
plot_panel_trxsh(3.00, 8,1,'st','d','rrr',[ 6 8 4 ],'lds','uaub','../results/epyc/20200302/mnkt100000_st','epyc','MKL','octave'); close; clear all;



















% Scratchpad
% st d
plot_panel_trxsh(3.80,16,1,'st','d','rrr',[ 6 8 4 ],'lds','uaub','../output_st/d','kbl','MKL','octave');
plot_panel_trxsh(3.80,16,1,'st','d','ccc',[ 6 8 4 ],'lds','uaub','../output_st/d','kbl','MKL','octave');
% mt d
plot_panel_trxsh(3.80,16,4,'mt','d','rrr',[ 6 8 10 ],'lds','uaub','../output_mt/d','kbl','MKL','octave');
plot_panel_trxsh(3.80,16,4,'mt','d','ccc',[ 6 8 10 ],'lds','uaub','../output_mt/d','kbl','MKL','octave');
% st s
plot_panel_trxsh(3.80,16,1,'st','s','rrr',[ 6 16 4 ],'lds','uaub','../output_st/s','kbl','MKL','octave');
plot_panel_trxsh(3.80,16,1,'st','s','ccc',[ 6 16 4 ],'lds','uaub','../output_st/s','kbl','MKL','octave');
% mt s
plot_panel_trxsh(3.80,16,4,'mt','s','rrr',[ 6 16 10 ],'lds','uaub','../output_mt/s','kbl','MKL','octave');
plot_panel_trxsh(3.80,16,4,'mt','s','ccc',[ 6 16 10 ],'lds','uaub','../output_mt/s','kbl','MKL','octave');

% tx2
plot_panel_4x5(2.20,8,1, 'st','../results/tx2/20190205/st',    'tx2',       'ARMPL'); close; clear all;
plot_panel_4x5(2.20,8,28,'1s','../results/tx2/20190205/jc4ic7','tx2_jc4ic7','ARMPL'); close; clear all;
plot_panel_4x5(2.20,8,56,'2s','../results/tx2/20190205/jc8ic7','tx2_jc8ic7','ARMPL'); close; clear all;

% skx
% pre-eigen:
%plot_panel_4x5(2.00,32,1, 'st','../results/skx/20190306/st',     'skx',        'MKL'); close; clear all;
%plot_panel_4x5(2.00,32,26,'1s','../results/skx/20190306/jc2ic13','skx_jc2ic13','MKL'); close; clear all;
%plot_panel_4x5(2.00,32,52,'2s','../results/skx/20190306/jc4ic13','skx_jc4ic13','MKL'); close; clear all;
% with eigen:
plot_panel_4x5(2.00,32,1, 'st','../results/skx/merged20190306_0328/st',     'skx',        'MKL',1); close; clear all;
plot_panel_4x5(2.00,32,26,'1s','../results/skx/merged20190306_0328/jc2ic13','skx_jc2ic13','MKL',1); close; clear all;
plot_panel_4x5(2.00,32,52,'2s','../results/skx/merged20190306_0328/jc4ic13','skx_jc4ic13','MKL',1); close; clear all;

% has
% pre-eigen:
%plot_panel_4x5(3.25,16,1, 'st','../results/has/20190206/st',       'has',          'MKL',1); close; clear all;
%plot_panel_4x5(3.00,16,12,'1s','../results/has/20190206/jc2ic3jr2','has_jc2ic3jr2','MKL',1); close; clear all;
%plot_panel_4x5(3.00,16,24,'2s','../results/has/20190206/jc4ic3jr2','has_jc4ic3jr2','MKL',1); close; clear all;
% with eigen:
plot_panel_4x5(3.25,16,1, 'st','../results/has/merged20190206_0328/st',       'has',          'MKL',1); close; clear all;
plot_panel_4x5(3.00,16,12,'1s','../results/has/merged20190206_0328/jc2ic3jr2','has_jc2ic3jr2','MKL',1); close; clear all;
plot_panel_4x5(3.00,16,24,'2s','../results/has/merged20190206_0328/jc4ic3jr2','has_jc4ic3jr2','MKL',1); close; clear all;

% epyc
% pre-eigen:
%plot_panel_4x5(3.00,8,1, 'st','../results/epyc/merged201903_0619/st','epyc',          'MKL'); close; clear all;
%plot_panel_4x5(2.55,8,32,'1s','../results/epyc/merged201903_0619/jc1ic8jr4','epyc_jc1ic8jr4','MKL'); close; clear all;
%plot_panel_4x5(2.55,8,64,'2s','../results/epyc/merged201903_0619/jc2ic8jr4','epyc_jc2ic8jr4','MKL'); close; clear all;
% with eigen:
plot_panel_4x5(3.00,8,1, 'st','../results/epyc/merged20190306_0319_0328/st',       'epyc',          'MKL',1); close; clear all;
plot_panel_4x5(2.55,8,32,'1s','../results/epyc/merged20190306_0319_0328/jc1ic8jr4','epyc_jc1ic8jr4','MKL',1); close; clear all;
plot_panel_4x5(2.55,8,64,'2s','../results/epyc/merged20190306_0319_0328/jc2ic8jr4','epyc_jc2ic8jr4','MKL',1); close; clear all;


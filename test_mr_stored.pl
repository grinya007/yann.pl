#!/usr/bin/env perl
use strict;
use warnings;
use experimental 'state', 'say';

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Layer;
use AI::YANN::Model::Regression;
use JSON::XS qw/decode_json/;
use PDL;

my $model = AI::YANN::Model::Regression->thaw(join('', `cat $ARGV[0]`));

my $tests = batches(1000, 5, $ARGV[1]);
for my $test (@$tests) {
  print "\n", $model->predict($test->[0])->flat(), "\n", $test->[1]->flat(), "\n";
}
print "\n";


sub batches {
  my ($m, $n, $file) = @_;

  state $f = {};
  unless ($f->{$file}) {
    open($f->{$file}, "cat $file |");
    #$f->{$file}->getline();
  }

  my @batches;
  OUTER: for my $i (1 .. $m) {
    my (@xs, @y);
    for my $j (1 .. $n) {
      my $l = $f->{$file}->getline();
      last OUTER unless $l;
      chomp($l);
      my ($xs, $y) = decode($l);
      for my $k (0 .. $#$xs) {
        $xs[$k] ||= [];
        push(@{ $xs[$k] }, $xs->[$k]);
      }
      push(@y, $y);
    }
    push(@batches, [[ map { pdl($_) } @xs ], pdl(\@y)]);
  }
  return \@batches;
}

sub decode {
  my ($line) = @_;
  my %l;
  @l{qw/
    connections_count
    has_fb
    hydro_ps_count
    hydro_res_count
    new_builds_count
    plants_count
    ramp_rates_n
    responses_count
    step_length_n
    steps
    storage_modelling_n
    unit_commitment
    wheeling_costs
    zones_count
    is_parallel
    stage
    is_sens
    ramp_rates
    storage_modelling
    uc
    step_length
    step_count
    step_unit
    country_ids
    aggregation
    memory_usage
  /} = split /:/, $line;

  my @x0;
  push(@x0, $l{'is_parallel'});

  state $stage = {
    'initial' => 0,
    'model'   => 1, 
    'final'   => 2,
  };
  my @stage_oh = (0) x scalar(keys %$stage);
  $stage_oh[$stage->{$l{'stage'}}] = 1 if $l{'stage'};
  push(@x0, @stage_oh);

  push(@x0, @l{qw/is_sens ramp_rates storage_modelling uc has_fb wheeling_costs/});

  state $step = {
    '24'    => 0,
    '168'   => 1,
    'month' => 2,
  };
  my @step_oh = (0) x scalar(keys %$step);
  $step_oh[$step->{$l{'step_unit'}}] = 1;
  push(@x0, @step_oh);

  state $agg = {
    'None'        => 0,
    'Main_focus'  => 1,
    'Custom'      => 2,
    'Medium'      => 3,
    'High'        => 4,
  };
  my @agg_oh = (0) x scalar(keys %$agg);
  $agg_oh[$agg->{$l{'aggregation'}}] = 1;
  push(@x0, @agg_oh);

  my @x1;
  state $countries = {};
  my $cids = decode_json($l{'country_ids'});
  for my $cid (@$cids) {
    $countries->{$cid} //= scalar(keys %$countries);
  }
  my @countries_oh = (0) x 200;
  @countries_oh[@$countries{@$cids}] = (1) x scalar(@$cids);
  push(@x1, @countries_oh);

  my @x2;
  push(@x2, $l{'step_length'} / 100);
  push(@x2, $l{'step_count'} / 100);
  push(@x2, $l{'plants_count'} / 1000);
  push(@x2, $l{'zones_count'} / 100);
  push(@x2, $l{'connections_count'} / 100);
  push(@x2, $l{'responses_count'} / 100);

  #         18     200   5
  return [ \@x0, \@x1, \@x2 ], [ $l{'memory_usage'} / 1024 ** 3 ];
}

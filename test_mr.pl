#!/usr/bin/env perl
use strict;
use warnings;
use experimental 'state', 'say';

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Model::Regression;
use JSON::XS qw/decode_json/;
use PDL;

my $model = AI::YANN::Model::Regression->new(
  'layers'  => [
    { 'output_size' => 16, 'input_size' => 218, 'activation' => 'sigmoid' },
    #{ 'output_size' => 128, 'activation' => 'sigmoid' },
    #{ 'output_size' => 64, 'activation' => 'relu' },
    #{ 'output_size' => 64, 'activation' => 'relu' },
    #{ 'output_size' => 64, 'activation' => 'relu' },
    { 'output_size' => 16, 'activation' => 'sigmoid' },
    { 'output_size' => 16, 'activation' => 'sigmoid' },
    { 'output_size' => 1 },
  ],
  'lr'      => 0.1,
);

my $b = batches(7, 100, $ARGV[1]);
my @train = @$b[0 .. $#$b - 1];
my @val = @$b[$#$b - 1 .. $#$b];

my $epochs = 300;
for my $epoch (1 .. $epochs) {
  my @losses;
  for my $batch (@train) {
    my $loss = $model->fit(@$batch);
    push(@losses, $loss);
  }
  my @vlosses;
  for my $batch (@val) {
    my $loss = $model->validate(@$batch);
    push(@vlosses, $loss);
  }
  print "$epoch/$epochs loss: ", pdl(\@losses)->avg(), " val_loss: ", pdl(\@vlosses)->avg(), "\n";
}

my $tests = batches(3, 5, $ARGV[1]);
for my $test (@$tests) {
  print $model->predict($test->[0])->flat(), "\n", $test->[1]->flat(), "\n\n";
}

sub batches {
  my ($m, $n, $file, $tail) = @_;

  state $f = {};
  unless ($f->{$file}) {
    open($f->{$file}, "zcat $file |". ($tail ? ' tail -n '.($m*$n).' |' : ''));
    $f->{$file}->getline();
  }

  my @batches;
  OUTER: for my $i (1 .. $m) {
    my (@x, @y);
    for my $j (1 .. $n) {
      my $l = $f->{$file}->getline();
      last OUTER unless $l;
      chomp($l);
      my ($x, $y) = decode($l);
      push(@x, $x);
      push(@y, $y);
    }
    push(@batches, [pdl(\@x), pdl(\@y)]);
  }
  return \@batches;
}

sub decode {
  my ($line) = @_;
  my %l;
  @l{qw/
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

  my @x;
  push(@x, $l{'is_parallel'});

  state $stage = {
    'initial' => 0,
    'model'   => 1, 
    'final'   => 2,
  };
  my @stage_oh = (0) x scalar(keys %$stage);
  $stage_oh[$stage->{$l{'stage'}}] = 1 if $l{'stage'};
  push(@x, @stage_oh);

  push(@x, @l{qw/is_sens ramp_rates storage_modelling uc/});

  push(@x, $l{'step_length'} / 100);
  push(@x, $l{'step_count'} / 1000);

  state $step = {
    '24'    => 0,
    '168'   => 1,
    'month' => 2,
  };
  my @step_oh = (0) x scalar(keys %$step);
  $step_oh[$step->{$l{'step_unit'}}] = 1;
  push(@x, @step_oh);

  state $countries = {};
  my $cids = decode_json($l{'country_ids'});
  for my $cid (@$cids) {
    $countries->{$cid} //= scalar(keys %$countries);
  }
  my @countries_oh = (0) x 200;
  @countries_oh[@$countries{@$cids}] = (1) x scalar(@$cids);
  push(@x, @countries_oh);

  state $agg = {
    'None'        => 0,
    'Main_focus'  => 1,
    'Custom'      => 2,
    'Medium'      => 3,
    'High'        => 4,
  };
  my @agg_oh = (0) x scalar(keys %$agg);
  $agg_oh[$agg->{$l{'aggregation'}}] = 1;
  push(@x, @agg_oh);

  return \@x, [ $l{'memory_usage'} / 1024 ** 3 ];
}

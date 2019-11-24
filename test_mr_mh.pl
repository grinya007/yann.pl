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

my $lb = AI::YANN::Layer->builder();
my $model = AI::YANN::Model::Regression->new(
  'lr'            => 0.01,
  'optimizer'     => 'adagrad',
  'output'        => $lb->('linear', 1,
    $lb->('relu', 256,
      $lb->('relu', 256,
        $lb->('relu', 256,
          $lb->('relu', 64,

            $lb->('relu', 64, 16, 0),
            $lb->('relu', 16, 2, 2),

          ),
          #$lb->('sigmoid', 256,

            #$lb->('sigmoid', 64, 16, 0),
            #$lb->('sigmoid', 16, 2, 2),

          #),
        ),

        # countries
        $lb->('relu', 128,
          $lb->('relu', 128,
            $lb->('relu', 256, 200, 1),
          ),
          #$lb->('sigmoid', 128,
            #$lb->('sigmoid', 256, 200, 1),
          #),
        ),
      ),
    ),
  ),
);

#use Data::Dumper;
#say Dumper $model;
#exit;

my $b = batches(50, 50, $ARGV[0]);
my @train = @$b[0 .. $#$b - 5];
my @val = @$b[$#$b - 4 .. $#$b];

my $epochs = 40;
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

my $tests = batches(10, 10, $ARGV[0]);
for my $test (@$tests) {
  print "\n", $model->predict($test->[0])->flat(), "\n", $test->[1]->flat(), "\n";
}
print "\n";

sub batches {
  my ($m, $n, $file, $tail) = @_;

  state $f = {};
  unless ($f->{$file}) {
    open($f->{$file}, "zcat $file |". ($tail ? ' tail -n '.($m*$n).' |' : ''));
    $f->{$file}->getline();
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

  my @x1;
  push(@x0, @l{qw/is_sens ramp_rates storage_modelling uc/});

  my @x2;
  push(@x2, $l{'step_length'} / 100);
  push(@x2, $l{'step_count'} / 1000);

  state $step = {
    '24'    => 0,
    '168'   => 1,
    'month' => 2,
  };
  my @step_oh = (0) x scalar(keys %$step);
  $step_oh[$step->{$l{'step_unit'}}] = 1;
  push(@x0, @step_oh);

  my @x3;
  state $countries = {};
  my $cids = decode_json($l{'country_ids'});
  for my $cid (@$cids) {
    $countries->{$cid} //= scalar(keys %$countries);
  }
  my @countries_oh = (0) x 200;
  @countries_oh[@$countries{@$cids}] = (1) x scalar(@$cids);
  push(@x1, @countries_oh);

  my @x4;
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

  #         16     200   2
  return [ \@x0, \@x1, \@x2 ], [ $l{'memory_usage'} / 1024 ** 3 ];
}

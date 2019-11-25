#!/usr/bin/env perl
use strict;
use warnings;

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Layer;
use AI::YANN::Model::Classification;
use PDL;

my $lb = AI::YANN::Layer->builder();
my $model = AI::YANN::Model::Classification->new(
  'lr'            => 0.01,
  'optimizer'     => 'adagrad',
  'output'        => $lb->('softmax', 10,
    $lb->('relu', 256,
      $lb->('relu', 512,
        $lb->('relu', 512, 28*28),
      ),
    ),
  ),
);

my $b = batches(50, 100, 255);
my @train = @$b[0 .. $#$b - 5];
my @val = @$b[$#$b - 5 .. $#$b];

my $epochs = 10;
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

my $t = batches(7520, 1, 1);
for (1 .. 1) {
  print "\n", "=" x 100, "\n";
  my $p = $model->predict($t->[-$_][0] / 255);
  print $t->[-$_][0]->reshape(28, 28);
  print $t->[-$_][1], $p;
}

sub batches {
  my ($m, $n, $f) = @_;
  open(my $images, "zcat $ARGV[0] |");
  open(my $labels, "zcat $ARGV[1] |");

  $images->read(my $ibuf, 16);
  $labels->read(my $lbuf, 8);

  my @batches;
  for my $i (1 .. $m) {
    my (@x, @y);
    for my $j (1 .. $n) {
      $images->read(my $ibuf, 28*28);
      $labels->read(my $lbuf, 1);
      push(@x, float([unpack('C*', $ibuf)]) / $f);
      push(@y, [unpack('C', $lbuf)]);
    }
    push(@batches, [pdl(\@x)->transpose(), pdl(\@y)]);
  }
  return \@batches;
}

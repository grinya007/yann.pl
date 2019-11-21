#!/usr/bin/env perl
use strict;
use warnings;

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Model::Classification;
use PDL;

my $model = AI::YANN::Model::Classification->new(
  'layers'  => [
    { 'output_size' => 1000, 'input_size' => 28*28, 'activation' => 'relu' },
    #{ 'output_size' => 1000, 'activation' => 'relu' },
    { 'output_size' => 500, 'activation' => 'relu' },
    #{ 'output_size' => 500, 'activation' => 'relu' },
    { 'output_size' => 10, 'activation' => 'softmax' },
  ],
  'lr'      => 0.1,
);

my $b = batches(50, 50, 255);

my $epochs = 10;
for my $epoch (1 .. $epochs) {
  my @losses;
  for my $batch (@$b) {
    my $loss = $model->fit(@$batch);
    push(@losses, $loss);
  }
  print "$epoch/$epochs ", pdl(\@losses)->avg(), "\n";
}

my $t = batches(2520, 1, 1);
for (1 .. 20) {
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
    push(@batches, [pdl(\@x), pdl(\@y)]);
  }
  return \@batches;
}

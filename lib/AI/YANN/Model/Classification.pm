package AI::YANN::Model::Classification;
use strict;
use warnings;

use base 'AI::YANN::Model';

sub _loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y->transpose()) .= 1;
  #print $y_hat, $one_hot, "\n\n";
  return -($one_hot->transpose() * ($y_hat + 1e-8)->log())->sum() / $y_hat->dim(0);
}

sub _d_loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y->transpose()) .= 1;
  return $y_hat - $one_hot->transpose();
}

1;

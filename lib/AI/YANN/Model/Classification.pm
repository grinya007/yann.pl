package AI::YANN::Model::Classification;
use strict;
use warnings;

use base 'AI::YANN::Model';

sub _loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y) .= 1;
  return -($one_hot->transpose() * ($y_hat + 1e-8)->log())->sum() / $y_hat->dim(0);
}

sub _d_loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y) .= 1;
  $one_hot = $one_hot->transpose();
  return $y_hat - $one_hot;
}

1;

package AI::YANN::Model::Regression;
use strict;
use warnings;

use base 'AI::YANN::Model';

sub _loss {
  my ($self, $y_hat, $y) = @_;
  return (($y_hat - $y) ** 2)->avg();
}

sub _d_loss {
  my ($self, $y_hat, $y) = @_;
  return (2 / $y->dim(0)) * ($y_hat - $y);
}

1;

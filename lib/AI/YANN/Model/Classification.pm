package AI::YANN::Model::Classification;
use strict;
use warnings;

use base 'AI::YANN::Model';

sub _loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros();
  $one_hot->index($y) .= 1;
  return -($one_hot * ($y_hat + 1e-8)->log())->sum() / $y_hat->dim(1);
}

sub _d_loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros();
  $one_hot->index($y) .= 1;
  return $one_hot;
}

1;

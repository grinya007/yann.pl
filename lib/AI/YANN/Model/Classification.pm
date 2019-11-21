package AI::YANN::Model::Classification;
use strict;
use warnings;

use base 'AI::YANN::Model';

sub _loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y) .= 1;
  #print $y_hat, $one_hot, "\n\n";
  return -($one_hot->transpose() * ($y_hat + 1e-8)->log())->sum() / $y_hat->dim(0);
}

sub _d_loss {
  my ($self, $y_hat, $y) = @_;
  my $one_hot = $y_hat->zeros()->transpose();
  $one_hot->index($y) .= 1;
  #print $y_hat, $one_hot;
  $one_hot = $one_hot->transpose();
  #my $c = $one_hot / ($y_hat + 1e-8) - (1 - $one_hot) / (1 - $y_hat + 1e-8);
  my $c = $y_hat - $one_hot;
  #print $c;
  return $c;
  # - (np.divide(Y, Y_hat + eps) - np.divide(1 - Y, 1 - Y_hat + eps))
}

1;

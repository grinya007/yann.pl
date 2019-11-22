package AI::YANN::Model;
use strict;
use warnings;

use AI::YANN::Layer;

sub new {
  my ($class, %args) = @_;
  
  my $prev_out_size;
  my @layers = @{ $args{'layers'} };
  for my $i (0 .. $#layers) {
    $layers[$i]{'input_size'} = $prev_out_size if $prev_out_size;
    $prev_out_size = $layers[$i]{'output_size'};
    $layers[$i] = AI::YANN::Layer->from_hash($layers[$i]);
  }

  return bless({
    '_layers' => \@layers,
    '_lr'     => $args{'lr'} // 0.001,
  }, $class);
}

sub fit {
  my ($self, $x, $y) = @_;
  
  my $y_hat = $x->transpose();
  for my $layer (@{ $self->{'_layers'} }) {
    $y_hat = $layer->forward($y_hat);
  }

  $y = $y->transpose();
  my $loss = $self->_loss($y_hat, $y);

  my $d_y_hat = $self->_d_loss($y_hat, $y);
  for my $layer (reverse @{ $self->{'_layers'} }) {
    $d_y_hat = $layer->backward($d_y_hat);
  }

  for my $layer (@{ $self->{'_layers'} }) {
    $layer->update($self->{'_lr'});
    $layer->zero_gradients();
    $layer->clean_up();
  }

  return $loss;
}

sub validate {
  my ($self, $x, $y) = @_;
  my $y_hat = $self->predict($x);
  return $self->_loss($y_hat, $y);
}

sub predict {
  my ($self, $x) = @_;
  my $y_hat = $x->transpose();
  for my $layer (@{ $self->{'_layers'} }) {
    $y_hat = $layer->forward($y_hat);
  }
  return $y_hat;
}

sub _loss   { ... }
sub _d_loss { ... }

1;

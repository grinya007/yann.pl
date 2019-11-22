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
    '_layers'     => \@layers,
    '_optimizer'  => $args{'optimizer'} // 'gradient_descent',
    '_lr'         => $args{'lr'} // 0.01,
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

  $self->update();

  return $loss;
}

sub validate {
  my ($self, $x, $y) = @_;
  my $y_hat = $self->predict($x);
  return $self->_loss($y_hat, $y->transpose());
}

sub predict {
  my ($self, $x) = @_;
  my $y_hat = $x->transpose();
  for my $layer (@{ $self->{'_layers'} }) {
    $y_hat = $layer->forward($y_hat);
  }
  return $y_hat;
}

sub update {
  my ($self) = @_;
  for my $layer (@{ $self->{'_layers'} }) {
    for my $param (@{ $layer->parameters() }) {
      if ($self->{'_optimizer'} eq 'adagrad') {
        $self->_adagrad($param);
      }
      else {
        $self->_gradient_descent($param);
      }
    }
    $layer->clean_up();
  }
}

sub _gradient_descent {
  my ($self, $param) = @_;
  $param->add_value(-$self->{'_lr'} * $param->gradient);
}

sub _adagrad {
  my ($self, $param) = @_;
  my $g = $param->gradient();
  $param->add_momentum($g ** 2);
  $param->add_value(
    (-$self->{'_lr'} * $g) / ($param->momentum() + 1e-8) ** 0.5
  );
}

sub _loss   { ... }
sub _d_loss { ... }

1;

package AI::YANN::Model;
use strict;
use warnings;

use AI::YANN::Layer;
use PDL;

sub new {
  my ($class, %args) = @_;
  return bless({
    '_output'     => $class->_create_layers($args{'output'}),
    '_optimizer'  => $args{'optimizer'} // 'gradient_descent',
    '_lr'         => $args{'lr'} // 0.01,
  }, $class);
}

sub _create_layers {
  my ($class, $hash) = @_;
  if ($hash->{'input_layers'}) {
    $hash->{'input_size'} = 0;
    for (@{ $hash->{'input_layers'} }) {
      $hash->{'input_size'} += $_->{'output_size'};
      $_ = $class->_create_layers($_);
    }
  }
  return AI::YANN::Layer->from_hash($hash);
}

sub fit {
  my ($self, $x, $y) = @_;
  
  my $y_hat = $self->_forward(
    $self->{'_output'}, $x->transpose()
  );

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
  return $self->_forward(
    $self->{'_output'}, $x->transpose()
  );
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

sub _forward {
  my ($self, $layer, $x) = @_;
  my $input = $x->transpose();
  if (my $input_layers = $layer->input_layers()) {
    $input = null();
    for (@$input_layers) {
      $input = $input->append(
        $self->_forward($_, $x)->transpose()
      );
    }
  }
  return $layer->forward($input->transpose());
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

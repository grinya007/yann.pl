package AI::YANN::Model;
use strict;
use warnings;

use AI::YANN::Layer;
use PDL;

sub new {
  my ($class, %args) = @_;
  return bless({
    '_output'     => $args{'output'},
    '_optimizer'  => $args{'optimizer'} // 'gradient_descent',
    '_lr'         => $args{'lr'} // 0.01,
  }, $class);
}

sub freeze {
  my ($self) = @_;
  return pack(
    '(L/a)2d',
    $self->{'_output'}->freeze(),
    $self->{'_optimizer'},
    $self->{'_lr'},
  );
}

sub thaw {
  my ($class, $blob) = @_;
  my ($output, $optimizer, $lr) = unpack('(L/a)2d', $blob);
  return $class->new(
    'output'      => AI::YANN::Layer->thaw($output),
    'optimizer'   => $optimizer,
    'lr'          => $lr,
  );
}

sub fit {
  my ($self, $x, $y) = @_;
  
  my $y_hat = $self->_forward($self->{'_output'}, $x);

  $y = $y->transpose();
  my $loss = $self->_loss($y_hat, $y);

  my $d_loss = $self->_d_loss($y_hat, $y);
  $self->_backward($self->{'_output'}, $d_loss);

  $self->_update($self->{'_output'});

  return $loss;
}

sub validate {
  my ($self, $x, $y) = @_;
  my $y_hat = $self->predict($x);
  return $self->_loss($y_hat, $y->transpose());
}

sub predict {
  my ($self, $x) = @_;
  return $self->_forward($self->{'_output'}, $x);
}

sub _update {
  my ($self, $layer) = @_;
  for my $param (@{ $layer->parameters() }) {
    if ($self->{'_optimizer'} eq 'adagrad') {
      $self->_adagrad($param);
    }
    else {
      $self->_gradient_descent($param);
    }
  }
  $layer->clean_up();
  if (my $input_layers = $layer->input_layers()) {
    for (@$input_layers) {
      $self->_update($_);
    }
  }
}

sub _forward {
  my ($self, $layer, $x) = @_;
  my $input = $x;
  if (my $input_layers = $layer->input_layers()) {
    $input = null();
    for (@$input_layers) {
      $input = $input->append(
        $self->_forward($_, $x)->transpose()
      );
    }
    $input = $input->transpose();
  }
  return $layer->forward($input);
}

sub _backward {
  my ($self, $layer, $d_loss) = @_;
  $d_loss = $layer->backward($d_loss);
  if (my $input_layers = $layer->input_layers()) {
    my $slice_from = 0;
    for (@$input_layers) {
      $self->_backward(
        $_, $d_loss->transpose()->slice(
          sprintf('%d:%d', $slice_from, $slice_from + $_->output_size() - 1)
        )->transpose());
      $slice_from += $_->output_size();
    }
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

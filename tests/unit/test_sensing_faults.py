from simulation.fault_injection.sensing_faults import SensingFaultInjector


def test_measurement_without_fault():
    injector = SensingFaultInjector()
    result = injector.apply(10.0)

    assert result.value == 10.0
    assert result.fault == "none"


def test_sensor_bias():
    injector = SensingFaultInjector(bias=2.0)
    result = injector.apply(10.0)

    assert result.value == 12.0
    assert result.fault == "bias"


def test_sensor_noise_is_reproducible():
    first = SensingFaultInjector(noise_std=1.0, seed=42)
    second = SensingFaultInjector(noise_std=1.0, seed=42)

    first_result = first.apply(10.0)
    second_result = second.apply(10.0)

    assert first_result == second_result
    assert first_result.fault == "noise"


def test_negative_noise_is_rejected():
    try:
        SensingFaultInjector(noise_std=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative noise_std should raise ValueError.")
